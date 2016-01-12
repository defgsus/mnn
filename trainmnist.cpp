/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/21/2015</p>
*/

#include <fstream>
#include <iomanip>

#include "trainmnist.h"
#include "mnn/mnn.h"
#include "mnistset.h"
#include "printstate.h"
#include "generate_input.h"

struct TrainMnist::Private
{
    typedef float Float;

    void loadSet();
    void saveAllLayers(const std::string& postfix);
    void createNet();
    void clearErrorCount();
    void prepareExpectedOutput(std::vector<Float>& v, uint8_t label) const;
    void train();
    void trainLabelStep();
    /** Runs all test images and gathers errors */
    void testPerformance();
    /** Gets error and stats, returns label from net */
    int getLabelError(int label);

    void runInputApproximation();

    MnistSet trainSet, testSet;
    MNN::StackSerial<Float> net;
    std::vector<Float> bufIn, bufExp, bufOut, bufErr;
    std::vector<size_t> errorsPerClass;
    Float learnRate, error, error_min, error_max, error_sum;
    size_t epoch, error_count;
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
        trainSet.load("/home/defgsus/prog/DATA/mnist/train-labels.idx1-ubyte",
                      "/home/defgsus/prog/DATA/mnist/train-images.idx3-ubyte");
        testSet.load("/home/defgsus/prog/DATA/mnist/t10k-labels.idx1-ubyte",
                      "/home/defgsus/prog/DATA/mnist/t10k-images.idx3-ubyte");

        std::cout << "loaded mnist set: "
                  << trainSet.width() << "x" << trainSet.height()
                  << " x " << trainSet.numSamples()
                  << " (" << testSet.numSamples() << " testing)"
                  << std::endl;
    }
    catch (const mnist_exception& e)
    {
        std::cerr << e.what() << std::endl;
        return;
    }
}

void TrainMnist::Private::saveAllLayers(const std::string& postfix)
{
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
           numOut = 10;
    bufIn.resize(numIn);
    bufExp.resize(numOut);
    bufOut.resize(numOut);
    bufErr.resize(numOut);
    errorsPerClass.resize(numOut);

#if 1
    auto l1 = new MNN::Rbm<Float, MNN::Activation::Logistic>(numIn, 200);
    //auto l2 = new MNN::Rbm<Float, MNN::Activation::Logistic>(300, 200);
    //auto l3 = new MNN::Rbm<Float, MNN::Activation::Linear>(200, numOut, .01, true);
    auto l3 = new MNN::Rbm<Float, MNN::Activation::Logistic>(200, numOut, 1., true);
    learnRate = 0.9;

    // load previous
#if 0
    l1->loadTextFile("mnist_rbm_100h.txt");
#elif 0
    l1->loadTextFile("../rbm_layer_0_300h.txt");
    l2->loadTextFile("../rbm_layer_1_200h.txt");
#endif

    l1->setMomentum(.7);
//    l2->setMomentum(.8);
    l3->setMomentum(.9);
    net.add( l1 );
//    net.add( l2 );
    net.add( l3 );

    net.brainwash();

#if 0
    net.loadTextFile("../mnist_e500_stack.txt");
    //runInputApproximation();
    /*size_t num = net.layer(0)->numOut();
    auto ln = new MNN::Rbm<Float, MNN::Activation::Linear>(num, num, 0.01);
    MNN::initPassThrough(ln);
    net.insert(1, ln);*/
#endif

#elif 0
    // ----- deep convolution -------

    //typedef MNN::Activation::LinearRectified ConvAct;
    typedef MNN::Activation::Linear ConvAct;
    //typedef MNN::Activation::Tanh ConvAct;

    // first layer
    auto l1 = new MNN::Convolution<Float, ConvAct>(
                    trainSet.width(), trainSet.height(), 25, 25);
    l1->setMomentum(.9);
    net.add(l1);
    auto lprev = l1;
    for (int i = 0; i < 3; ++i)
    {
        size_t newSize = 2;//lprev->kernelWidth() / 3;
        if (newSize > lprev->scanWidth()
            || newSize == lprev->kernelWidth()
            || newSize < 2)
            break;
        auto lx = new MNN::Convolution<Float, ConvAct>(
                   lprev->scanWidth(), lprev->scanHeight(),
                   newSize, newSize);
        lx->setMomentum(.9);
        net.add(lx);
        lprev = lx;
    }
    auto lout = new MNN::Perceptron<Float, MNN::Activation::Linear>(net.numOut(), numOut);
    lout->setMomentum(.9);
    net.add(lout);
    //auto lout2 = new MNN::Perceptron<Float, MNN::Activation::Linear>(lout->numOut(), numOut, 0.1);
    //lout2->setMomentum(.9);
    //net.add(lout2);

    learnRate = 0.0001;

    net.brainwash(0.5);
#elif 0

    // ----- convolution -------

    //typedef MNN::Activation::LinearRectified ConvAct;
    typedef MNN::Activation::Linear ConvAct;
    //typedef MNN::Activation::Tanh ConvAct;

    auto l1 = new MNN::Convolution<Float, ConvAct>(
                    trainSet.width(), trainSet.height(), 10, 10);
    l1->setMomentum(.9);
    net.add(l1);
    auto lprev = l1;

    {
        auto l = lprev = new MNN::Convolution<Float, ConvAct>(
                            lprev->scanWidth(), lprev->scanHeight(),
                            9, 9);
        l->setMomentum(.9);
        net.add(l);
    }
    auto lout = new MNN::Perceptron<Float, MNN::Activation::Linear>(net.numOut(), numOut);
    lout->setMomentum(.9);
    net.add(lout);

    learnRate = 0.0001;

    net.brainwash(0.5);

#else
    // ---------- parallel convolution ------------

    typedef MNN::Activation::LinearRectified ConvAct;
    //typedef MNN::Activation::Tanh ConvAct;

    auto stack = new MNN::StackParallel<Float>;

    {
        auto l1 = new MNN::Convolution<Float, ConvAct>(trainSet.width(), trainSet.height(), 1, 1);
        l1->setMomentum(.9);
        stack->add(l1);
    }
    {
        auto l1 = new MNN::Convolution<Float, ConvAct>(trainSet.width(), trainSet.height(), 3, 3);
        l1->setMomentum(.9);
        stack->add(l1);
    }
    if (0)
    {
        auto l1 = new MNN::Perceptron<Float, ConvAct>(trainSet.width()*trainSet.height(), 50);
        l1->setMomentum(.9);
        stack->add(l1);
    }
    net.add(stack);
    if (0)
    {
        auto lout = new MNN::Perceptron<Float, MNN::Activation::Linear>(net.numOut(), 100);
        lout->setMomentum(.9);
        net.add(lout);
    }
    auto lout = new MNN::Perceptron<Float, MNN::Activation::Linear>(net.numOut(), numOut);
    lout->setMomentum(.9);
    net.add(lout);

    learnRate = 0.0005;

    net.brainwash(0.1);

#endif


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
        trainLabelStep();

        const int num = 5000;

        if (epoch % 50000 == 0)
        {
            if (learnRate > 0.000002)
                learnRate *= 0.5;
        }

        // test performance on validation set
        // once in a while
        if (epoch % 60000 == 0)
        {
            testPerformance();
#if 0
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
            else if (error_count < 1000)
                saveAllLayers("../mnist_e1000");
#endif
            clearErrorCount();
        }
        else
        if (epoch % num == 0)
        {
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
#if 1
            if (error_percent < 9)
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
    v[label] = .9;
}

void TrainMnist::Private::trainLabelStep()
{
    const size_t numBatch = 1;

    for (auto& b : bufErr)
        b = 0.;

    for (size_t batch = 0; batch < numBatch; ++batch, ++epoch)
    {
        // choose a sample
        size_t num = size_t(rand()) % trainSet.numSamples();
        uint8_t label = trainSet.label(num);
        const Float *image = trainSet.image(num);

        // prepare expected output
        prepareExpectedOutput(bufExp, label);

    #if 0
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
            e = e * e * e;
            bufErr[i] += e;
        }

        getLabelError(label);
    }

    // scale error by batch size
    for (auto& b : bufErr)
        b /= numBatch;

//    for (auto& e : bufErr)
//        e = e * (Float(1) - e);

    // learn
    net.bprop(&bufErr[0], NULL, learnRate);
}

int TrainMnist::Private::getLabelError(int label)
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

    auto& set = testSet;

    for (size_t num=0; num < set.numSamples(); ++num)
    {
        uint8_t label = set.label(num);
        const Float *image = set.image(num);

        net.fprop(image, &bufOut[0]);

        int answer = getLabelError(label);
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
}

void TrainMnist::Private::runInputApproximation()
{
    GenerateInput<Float> gen;

    std::vector<Float> output(net.numOut());
    prepareExpectedOutput(output, 8);
    gen.setExpectedOutput(&output[0], output.size());

    const size_t numIt = 1000;
    size_t it = 0;
    while (true)
    {
        gen.approximateInput(net, numIt);
        it += numIt;

        std::cout << "iteration " << it << ", error best " << gen.error()
                  << ", worst " << gen.errorWorst() << "\n";
        printStateAscii(gen.input(), trainSet.width(), trainSet.height());
        printState(gen.output(), net.numOut(), 1);
        std::cout << std::endl;
    }
}
