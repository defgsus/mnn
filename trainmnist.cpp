/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/21/2015</p>
*/

/** @page results
 *
 *  MNIST:
 *      784-500-500-10 PerceptronBias(TANH), last layer LINEAR
 *                  ~2.3% (1.2% on training set) after ~11,000,000 steps
 *
 *
 **/

// use cifar instead of mnist
//#define CIFAR

#include <fstream>
#include <iomanip>

#include "mnn/mnn.h"
#include "trainmnist.h"
#ifdef CIFAR
#   include "cifarset.h"
#else
#   include "mnistset.h"
#endif
#include "printstate.h"
#include "generate_input.h"

struct TrainMnist::Private
{
    typedef float Float;
#ifdef CIFAR
    typedef CifarSet DataSet;
#else
    typedef MnistSet DataSet;
#endif

    Private()
        : doTrainCD (false)
        , cdnet     (0)
    { }

    void loadSet();
    void saveAllLayers(const std::string& postfix);
    void createNet();
    void clearErrorCount();
    void prepareExpectedOutput(std::vector<Float>& v, uint8_t label) const;
    const Float* getImage(DataSet& set, uint32_t index) const;
    void train();
    void trainLabelStep();
    template <class Rbm>
    void trainCDStep(Rbm& rbm);
    template <class Net>
    void trainReconStep(Net& net);
    /** Runs all test images and gathers errors */
    void testPerformance();
    /** Gets error and stats, returns label from net */
    template <class Net>
    int getLabelError(const Net& net, int label);

    void runInputApproximation();

    DataSet trainSet, testSet;
    MNN::StackSerial<Float> net;
    std::vector<Float> bufIn, bufExp, bufOut, bufErr;
    std::vector<size_t> errorsPerClass;
    Float learnRate,
        error, error_min, error_max, error_sum;
    size_t numBatch, epoch, error_count;    
    int64_t saved_error_count;
    bool doTrainCD;
    MNN::ContrastiveDivergenceInterface<Float>* cdnet;
};

TrainMnist::TrainMnist()
    : p_        (new Private())
{

}

TrainMnist::~TrainMnist()
{
    delete p_;
}

void TrainMnist::exec()
{
    p_->loadSet();
    p_->createNet();

    p_->net.info();
    p_->train();
}

void TrainMnist::Private::loadSet()
{
    try
    {
#ifdef CIFAR
        trainSet.load("/home/defgsus/prog/DATA/cifar-10/data_batch_1.bin");
        trainSet.load("/home/defgsus/prog/DATA/cifar-10/data_batch_2.bin");
        trainSet.load("/home/defgsus/prog/DATA/cifar-10/data_batch_3.bin");
        trainSet.load("/home/defgsus/prog/DATA/cifar-10/data_batch_4.bin");
        trainSet.load("/home/defgsus/prog/DATA/cifar-10/data_batch_5.bin");
        testSet.load("/home/defgsus/prog/DATA/cifar-10/test_batch.bin");
#else
        trainSet.load("/home/defgsus/prog/DATA/mnist/train-labels.idx1-ubyte",
                      "/home/defgsus/prog/DATA/mnist/train-images.idx3-ubyte");
        trainSet.normalize();
        testSet.load("/home/defgsus/prog/DATA/mnist/t10k-labels.idx1-ubyte",
                      "/home/defgsus/prog/DATA/mnist/t10k-images.idx3-ubyte");
        testSet.normalize();
        /*
        for (int i=0; i<10; ++i)
            printStateAscii(trainSet.getTransformedImage(10, 4), trainSet.width(), trainSet.height());
        abort();
        */
#endif
        std::cout << "loaded set: "
                  << trainSet.width() << "x" << trainSet.height()
                  << " x " << trainSet.numSamples()
                  << " (" << testSet.numSamples() << " testing)"
                  << std::endl;
    }
    catch (const MNN::Exception& e)
    {
        std::cerr << e.what() << std::endl;
        return;
    }
}

const TrainMnist::Private::Float*
TrainMnist::Private::getImage(DataSet &set, uint32_t num) const
{
#if 0
    return set.image(num);
#elif 0
    const Float *image = set.getNoisyBackgroundImage(
                num, 0.4, MNN::rnd(-0.3, 0.3), MNN::rnd(0., 1.));
    //printStateAscii(image, trainSet.width(), trainSet.height());
    return image;
#elif 1
    return set.getTransformedImage(num, MNN::rnd(0.f, 4.f));
#else
    const Float *image = set.getNoisyImage(
                num, MNN::rnd(-0.3, 0.), MNN::rnd(0., .3));
    return image;
#endif
}

void TrainMnist::Private::saveAllLayers(const std::string& postfix)
{
    std::cout << "saving '" << postfix << "'" << std::endl;
    net.saveTextFile(postfix + "_stack.txt");
    for (size_t i=0; i<net.numLayer(); ++i)
    {
        std::stringstream str;
        str << postfix << "_layer_" << i
            << "_" << net.layer(i)->numOut() << "h.txt";
        net.layer(i)->saveTextFile(str.str());
    }
}

void TrainMnist::Private::createNet()
{
    size_t numIn = trainSet.width() * trainSet.height(),
           numOut = trainSet.numClasses();
    bufIn.resize(numIn);
    bufExp.resize(numOut);
    bufOut.resize(numOut);
    bufErr.resize(numOut);
    errorsPerClass.resize(numOut);

    saved_error_count = -1;
    numBatch = 1;
    doTrainCD = false;

#if 0
    // --- load autoencoder stack ----
    for (int i=0; i<5; ++i)
    {
        auto l = new MNN::Perceptron<Float, MNN::Activation::Linear>(1, 1, 1., false);
        l->setMomentum(.5);
        net.add(l);
    }
    net.loadTextFile("../autoencoder-stack-5-mnist-14x14.txt");
    trainSet.scale(14, 14); testSet.scale(14, 14);
    if (net.numOut() != numOut)
    {
        auto l = new MNN::Perceptron<Float, MNN::Activation::Linear>(net.numOut(), numOut, 1.);
        l->setMomentum(.5);
        l->brainwash(0.1);
        net.add(l);
    }
    learnRate = .01;

#elif 0
    auto l1 = new MNN::Rbm<Float, MNN::Activation::Linear>(numIn, 200);
    //auto l2 = new MNN::Rbm<Float, MNN::Activation::Logistic>(300, 200);
    //auto l3 = new MNN::Rbm<Float, MNN::Activation::Linear>(200, numOut, .01, true);
    auto l3 = new MNN::Rbm<Float, MNN::Activation::Linear>(200, numOut, 1., true);
    learnRate = 0.09;

    // load previous
#if 0
    l1->loadTextFile("mnist_rbm_100h.txt");
#elif 0
    l1->loadTextFile("../rbm_layer_0_300h.txt");
    //l2->loadTextFile("../rbm_layer_1_200h.txt");
#endif

    l1->setMomentum(.7);
//    l2->setMomentum(.8);
    l3->setMomentum(.9);
    net.add( l1 );
//    net.add( l2 );
    net.add( l3 );

    net.brainwash(0.1);
    learnRate = 0.01;
    //cdnet = l1; doTrainCD = true;

#if 0
    net.loadTextFile("../mnist_e500_stack.txt");
    //runInputApproximation();
    /*size_t num = net.layer(0)->numOut();
    auto ln = new MNN::Rbm<Float, MNN::Activation::Linear>(num, num, 0.01);
    MNN::initPassThrough(ln);
    net.insert(1, ln);*/
#endif

#elif 1
    // ----- classic dense matrix -----

    typedef MNN::Activation::Tanh Act;
    //typedef MNN::Activation::Logistic Act;
    {
        auto l = new MNN::PerceptronBias<Float, Act>(
                    trainSet.width() * trainSet.height(), 100);
        l->setMomentum(.9);
        //l->setDropOutMode(MNN::DO_TRAIN);
        //l->setDropOut(.2);
        //l->setLearnRateBias(.2);
        net.add(l);
    }
    if (1)
    {
        auto l = new MNN::PerceptronBias<Float, Act>(net.numOut(), 100);
        l->setMomentum(.9);
        //l->setDropOutMode(MNN::DO_TRAIN);
        //l->setDropOut(.5);
        //l->setLearnRateBias(.2);
        net.add(l);
    }
    // output layer
    {
        auto l = new MNN::PerceptronBias<Float, MNN::Activation::Linear>(net.numOut(), numOut);
        l->setMomentum(.9);
        l->setSoftmax(true);
        //l->setDropOutMode(MNN::DO_TRAIN);
        //l->setDropOut(.5);
        //l->setLearnRateBias(.2);
        net.add(l);
    }

    net.brainwash();
    learnRate = 0.001;
    numBatch = 1;

#elif 0

    // ----- classic dense matrix -----

    typedef MNN::Activation::Tanh Act;
    //typedef MNN::Activation::Logistic Act;
    {
        auto l = new MNN::PerceptronBias<Float, Act>(
                    trainSet.width() * trainSet.height(), 500);
        l->loadTextFile("../nets/tanh/autoencoder-mnist-noise-500h.txt");
        l->setMomentum(.9);
        l->setLearnRate(0.0);
        l->setLearnRateBias(0.0);
        //l->setDropOutMode(MNN::DO_TRAIN);
        //l->setDropOut(.2);
        net.add(l);
    }
    {
        auto l = new MNN::PerceptronBias<Float, Act>(net.numOut(), 500);
        l->loadTextFile("../nets/tanh/autoencoder-mnist-noise-l2-500v-500h.txt");
        l->setMomentum(.9);
        l->setLearnRate(0.0);
        l->setLearnRateBias(0.0);
        //l->setDropOutMode(MNN::DO_TRAIN);
        //l->setDropOut(.5);
        net.add(l);
    }
    // output layer
    {
        auto l = new MNN::PerceptronBias<Float, MNN::Activation::Linear>(net.numOut(), numOut);
        l->brainwash();
        l->setMomentum(.9);
        l->setLearnRateBias(.1);
        //l->setDropOutMode(MNN::DO_TRAIN);
        //l->setDropOut(.5);
        l->setSoftmax(true);
        net.add(l);
    }

    net.loadTextFile("../mnist_e300_stack.txt");
    for (size_t i=0; i<net.numLayer()-1; ++i)
    {
        MNN::setLearnRate(net.layer(i), Float(0.7));
        MNN::setLearnRateBias(net.layer(i), Float(0.05));
    }
    saved_error_count = 247;
    //net.brainwash();
    learnRate = 0.00001;
    numBatch = 1;

#elif 0
    // ----- deep convolution -------

    //typedef MNN::Activation::LinearRectified ConvAct;
    typedef MNN::Activation::Linear ConvAct;
    //typedef MNN::Activation::Tanh ConvAct;

    // first layer
    auto l1 = new MNN::Convolution<Float, ConvAct>(
                    trainSet.width(), trainSet.height(), 5, 5);
    l1->setMomentum(.9);
    net.add(l1);
    auto lprev = l1;
    for (int i = 0; i < 3; ++i)
    {
        size_t newSize = lprev->kernelWidth() * 1.3;
        if (newSize > lprev->scanWidth()
            || newSize == lprev->kernelWidth()
            || newSize < 2)
            break;
        auto lx = new MNN::Convolution<Float, ConvAct>(
                   lprev->scanWidth(), lprev->scanHeight(),
                   newSize, newSize);
        lx->setMomentum(.9);
        if (lx->scanWidth() * lx->scanHeight() < 50)
        {
            delete lx;
            break;
        }
        net.add(lx);
        lprev = lx;
    }
    auto lout = new MNN::Perceptron<Float, MNN::Activation::Linear>(net.numOut(), numOut);
    lout->setMomentum(.9);
    net.add(lout);
    //auto lout2 = new MNN::Perceptron<Float, MNN::Activation::Linear>(lout->numOut(), numOut, 0.1);
    //lout2->setMomentum(.9);
    //net.add(lout2);

    learnRate = 0.001;

    net.brainwash(1.);
#elif 0

    // ----- convolution -------

    //typedef MNN::Activation::LinearRectified ConvAct;
    typedef MNN::Activation::LinearRectified ConvAct;
    //typedef MNN::Activation::Tanh ConvAct;

    auto l1 = new MNN::Convolution<Float, ConvAct>(
                    trainSet.width(), trainSet.height(), 25, 25, 0.01);
    l1->setMomentum(.9);
    net.add(l1);
    auto lprev = l1;
    if (0)
    {
        auto l = lprev = new MNN::Convolution<Float, ConvAct>(
                            lprev->scanWidth(), lprev->scanHeight(),
                            9, 9);
        l->setMomentum(.9);
        net.add(l);
    }
    if (0)
    {
        auto l = new MNN::PerceptronBias<Float, ConvAct>(
                            lprev->scanWidth() * lprev->scanHeight(),
                            200);
        l->setMomentum(.9);
        net.add(l);
    }
    auto lout = new MNN::PerceptronBias<Float, MNN::Activation::Linear>(net.numOut(), numOut);
    lout->setMomentum(.9);
    lout->setSoftmax(true);
    net.add(lout);

    learnRate = 0.0006;

    net.brainwash(0.5);

#else
    // ---------- parallel convolution ------------

    typedef MNN::Activation::LinearRectified ConvAct;
    //typedef MNN::Activation::Tanh ConvAct;

    auto stack = new MNN::StackParallel<Float>;

    for (int i = 0; i<16; ++i)
    {
        auto l1 = new MNN::Convolution<Float, ConvAct>(trainSet.width()/4, trainSet.height()/4,
                                                       4, 4);
        l1->setMomentum(.9);
        stack->add(l1);
    }
    if (0)
    {
        auto l1 = new MNN::Convolution<Float, ConvAct>(trainSet.width(), trainSet.height(), 3, 3);
        l1->setMomentum(.9);
        stack->add(l1);
    }
    if (0)
    {
        auto l1 = new MNN::Perceptron<Float, ConvAct>(trainSet.width()*trainSet.height(),
                                                      50, 0.1);
        l1->setMomentum(.9);
        stack->add(l1);
    }
    net.add(stack);
    std::cout << net.numIn() << std::endl;


    // -- output layers --
    if (0)
    {
        auto lout = new MNN::Perceptron<Float, MNN::Activation::Linear>(net.numOut(), 100, 0.1);
        lout->setMomentum(.9);
        net.add(lout);
    }
    auto lout = new MNN::PerceptronBias<Float, MNN::Activation::Linear>(net.numOut(), numOut);
    lout->setMomentum(.9);
    lout->setSoftmax(true);
    net.add(lout);

    learnRate = 0.00001;

    net.brainwash(0.1);

#endif

    assert(net.numIn() == trainSet.width() * trainSet.height());
    assert(net.numOut() == 10);
}

void TrainMnist::Private::train()
{
    // --------- training ---------

    clearErrorCount();
    testPerformance();
    clearErrorCount();

//    bool doGrow = true;

    epoch = 0;
    while (true)
    {
        if (doTrainCD && cdnet)
            trainCDStep(*cdnet);
#if 0
        else if (auto rec = dynamic_cast<MNN::ReconstructionInterface<Float>*>(net.layer(0)))
        {
            trainReconStep(*rec);
        }
#endif
        else
            trainLabelStep();

        const int num = 5000;

#if 0
        // decrease learnrate
        if (epoch % 2000 == 0)
        {
            if (learnRate > 0.0002)
            {
                learnRate *= 0.9;
                //std::cout << "learnrate " << learnRate << std::endl;
            }
        }
#endif

#if 1
        // end unsupervised pre-training
        if (doTrainCD && epoch >= 60000)
        {
            doTrainCD = false;
            clearErrorCount();
            //net.layer(0)->saveTextFile("rbm_mnist_noisy_200.txt");
        }
#endif

        // test performance on validation set
        // once in a while
        if (epoch % 60000 == 0)
        {
            testPerformance();
#if 1
            if (saved_error_count < 0 || error_count < size_t(saved_error_count))
            {
                // save when error < x
                if (error_count < 100)
                    saveAllLayers("../mnist_e100");
                else if (error_count < 200)
                    saveAllLayers("../mnist_e200");
                else if (error_count < 300)
                    saveAllLayers("../mnist_e300");
                else if (error_count < 400)
                    saveAllLayers("../mnist_e400");
                else if (error_count < 500)
                    saveAllLayers("../mnist_e500");
                else if (error_count < 600)
                    saveAllLayers("../mnist_e600");
                else if (error_count < 1000)
                    saveAllLayers("../mnist_e1000");

                saved_error_count = error_count;
            }
#endif
            clearErrorCount();
        }
        else
        if (epoch % num == 0)
        {
            //printState(net.outputs(), 10, 1);

            Float error_percent = Float(error_count) / num * 100;

            std::cout << "epoch " << std::setw(7) << epoch
                      << ", error " << error_min
                      << " - " << error_max
                      << ", pc";
            for (size_t j=0; j<errorsPerClass.size(); ++j)
                std::cout << " " << std::setw(3) << errorsPerClass[j];
            std::cout << ", % " << error_percent
                      << ", avw " << net.getWeightAverage()
                      << std::endl;

            clearErrorCount();
#if 0
            if (epoch >= 120000
             && error_percent < 18
              && !doTrainCD)
                runInputApproximation();
#endif

#if 0
            if (error_percent < 20 && doGrow)
            {
                net.layer(0)->grow(net.layer(0)->numIn(), 1000, 0.05);
                net.updateLayers();
                net.info();
                doGrow = false;
            }
#endif
        }
    }
}

void TrainMnist::Private::clearErrorCount()
{
    error_count = 0;
    error_max = 0.;
    error_min = -1.;
    for (auto&x : errorsPerClass)
            x = 0.;
}

void TrainMnist::Private::prepareExpectedOutput(std::vector<Float>& v, uint8_t label) const
{
    for (auto& f : v)
        f = 0.0;
    v[label] = 1.;
}


void TrainMnist::Private::trainLabelStep()
{
    for (auto& b : bufErr)
        b = 0.;

    size_t num = size_t(rand()) % trainSet.numSamples();
    for (size_t batch = 0; batch < numBatch; ++batch, ++epoch)
    {
        // choose a sample
        num = trainSet.nextRandomSample(num);
        uint8_t label = trainSet.label(num);
        const Float* image = getImage(trainSet, num);

        // prepare expected output
        prepareExpectedOutput(bufExp, label);

    #if 1
        net.fprop(image, &bufOut[0]);
    #else // with noise
        for (size_t i=0; i<bufIn.size(); ++i)
            bufIn[i] = image[i] + MNN::rnd(-.1, .1);
        net.fprop(&bufIn[0], &bufOut[0]);
    #endif

        // calc error
        for (size_t i=0; i<bufExp.size(); ++i)
        {
            Float e = bufExp[i] - bufOut[i];
            //if (std::abs(e) > 0.05)
            //    e = e > 0. ? 1. : -1.;
            //e = std::max(Float(-1), std::min(Float(1), e));
            //e = e * e * e;
            bufErr[i] += e;
        }

        getLabelError(net, label);
    }

    // scale error by batch size
    for (auto& b : bufErr)
        b /= numBatch;

//    for (auto& e : bufErr)
//        e = e * (Float(1) - e);

    // learn
    net.bprop(&bufErr[0], NULL, learnRate);
}

template <class Net>
int TrainMnist::Private::getLabelError(const Net& net, int label)
{
    // get error from actual label number
    int answer = -1;
    Float ma = 0.;
    for (size_t i=0; i<net.numOut(); ++i)
    {
        Float o = net.output(i);
        if (o > ma)
        {
            ma = o;
            answer = i;
        }
    }

    // no label == 11, otherwise distance between
    // expected and performed digit
    error = answer < 0 ? 11. : std::abs(Float(answer - label));
    // count number of overall and per-class error
    if (answer != label)
    {
        ++error_count;
        ++errorsPerClass[label];
    }

    // -- get error stats --

    error_max = std::max(error_max, error);
    if (error_min < 0.)
        error_min = error;
    else
        error_min = std::min(error_min, error);
    error_sum += error;

    return answer;
}

void TrainMnist::Private::testPerformance()
{
    clearErrorCount();

    // select the test set
    auto& set = testSet;

    // get a net copy
    auto net = this->net.getCopy();
    // set dropout mode to PERFORM
    // Note: PERFORM mode is ignored for
    // networks that havn't been trained with dropout
    if (auto d = dynamic_cast<MNN::SetDropOutInterface<Float>*>(net))
        d->setDropOutMode(MNN::DO_PERFORM);

    for (size_t num=0; num < set.numSamples(); ++num)
    {
        uint8_t label = set.label(num);
        const Float *image = set.image(num);

        net->fprop(image, &bufOut[0]);

        int answer = getLabelError(*net, label);
        (void)answer;
#if 0
        if (error)
        {
            std::cout << "WRONG " << answer << " / " << (int)label << std::endl;
            printStateAscii(image, testSet.width(), testSet.height());
        }
#endif
    }

    // output performance
    std::cout << "validation,    "
              << "error " << error_min
              << " - " << error_max
              << ", pc";
    for (size_t j=0; j<errorsPerClass.size(); ++j)
        std::cout << " " << std::setw(3) << errorsPerClass[j];
    std::cout << ", E " << error_count
              << std::endl;

    delete net;
}

void TrainMnist::Private::runInputApproximation()
{
    // get a net copy
    auto net = this->net.getCopy();
    if (auto d = dynamic_cast<MNN::SetDropOutInterface<Float>*>(net))
        d->setDropOutMode(MNN::DO_PERFORM);

    GenerateInput<Float> gen(-0.1, 0.1);

    std::vector<Float> output(net->numOut());
    prepareExpectedOutput(output, 8);
    gen.setExpectedOutput(&output[0], output.size());

    const size_t numIt = 1000;
    size_t it = 0;
    while (true)
    {
        gen.approximateInput(*net, numIt);
        it += numIt;

        std::cout << "iteration " << it << ", error best " << gen.error()
                  << ", worst " << gen.errorWorst() << "\n";
        printStateAscii(gen.input(), trainSet.width(), trainSet.height());
        printState(gen.output(), net->numOut(), 1);
        std::cout << std::endl;
    }
}

template <class Rbm>
void TrainMnist::Private::trainCDStep(Rbm& rbm)
{
    // choose a sample
    uint32_t num = uint32_t(rand()) % trainSet.numSamples();
    uint8_t label = trainSet.label(num);
    const Float* image = getImage(trainSet, num);

    error = rbm.contrastiveDivergence(image, 1, learnRate);
    error *= 100.;

    ++epoch;

    if (error == 0 || error > 5)
    {
        ++error_count;
        ++errorsPerClass[label];
    }

    // -- get error stats --

    error_sum += error;
    error_max = std::max(error_max, error);
    if (error > 0.)
    {
        if (error_min < 0.)
            error_min = error;
        else
            error_min = std::min(error_min, error);
    }
}

template <class Net>
void TrainMnist::Private::trainReconStep(Net& net)
{
    // choose a sample
    uint32_t num = uint32_t(rand()) % trainSet.numSamples();
    uint8_t label = trainSet.label(num);
    const Float* image = getImage(trainSet, num);

    error = net.reconstructionTraining(image, learnRate * 0.001);
    error *= 100.;

    ++epoch;

    if (error == 0 || error > 5)
    {
        ++error_count;
        ++errorsPerClass[label];
    }

    // -- get error stats --

    error_sum += error;
    error_max = std::max(error_max, error);
    if (error > 0.)
    {
        if (error_min < 0.)
            error_min = error;
        else
            error_min = std::min(error_min, error);
    }

}
