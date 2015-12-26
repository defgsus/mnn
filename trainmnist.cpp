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

struct TrainMnist::Private
{
    void loadSet();
    void saveAllLayers(const std::string& postfix);
    void createNet();
    void clearErrorCount();
    void train();
    void trainLabelStep();
    /** Runs all test images and gathers errors */
    void testPerformance();
    void getLabelError(int label);

    typedef float Float;

    MnistSet trainSet, testSet;
    MNN::StackSerial<Float> net;
    std::vector<Float> bufExp, bufOut;
    std::vector<size_t> errorsPerClass;
    Float error, error_min, error_max, error_sum;
    size_t error_count;
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
    size_t numIn = trainSet.width() * trainSet.height();
    bufExp.resize(10);
    bufOut.resize(10);

    auto l1 = new MNN::Rbm<Float, MNN::Activation::Logistic>(numIn, 100);
    auto l2 = new MNN::Rbm<Float, MNN::Activation::Logistic>(100, 10);
    //auto l3 = new MNN::Rbm<Float, MNN::Activation::Logistic>(100, 10);

    net.brainwash();

    // load previous
#if 0
    l1->loadTextFile("mnist_rbm_100h.txt");
#elif 0
    l1->loadTextFile("rbm_layer_0.txt");
    l2->loadTextFile("rbm_layer_1.txt");
#endif

    l1->setMomentum(.5);
    l2->setMomentum(.9);
    //l2->setDropOut(MNN::DO_TRAIN);
    //l3->setMomentum(.9);
    net.add( l1 );
    net.add( l2 );
    //net.add( l3 );
}

void TrainMnist::Private::train()
{
    // --------- training ---------

    errorsPerClass.resize(10);
    clearErrorCount();

    for (int i=0; i<10000000; ++i)
    {
        trainLabelStep();

        const int num = 5000;

        // test performance on validation set
        // once in a while
        if (i % 60000 == 0)
        {
            testPerformance();
            // error < 10% ?
            if (error_count < 1000)
                saveAllLayers("mnist-best");

            clearErrorCount();
        }
        else
        if (i % num == 0)
        {
            std::cout << "epoch " << std::setw(7) << i
                      << ", error " << error_min
                      << " - " << error_max
                      << ", pc";
            for (size_t j=0; j<errorsPerClass.size(); ++j)
                std::cout << " " << std::setw(3) << errorsPerClass[j];
            std::cout << ", % " << (float(error_count) / num * 100)
                      << std::endl;

            clearErrorCount();
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

void TrainMnist::Private::trainLabelStep()
{
    // choose a sample
    size_t num = size_t(rand()) % trainSet.numSamples();
    uint8_t label = trainSet.label(num);
    const Float *image = trainSet.image(num);

    // prepare expected output
    for (auto& f : bufExp)
        f = 0.0;
    bufExp[label] = .9;

    net.fprop(image, &bufOut[0]);

    // calc error
    for (size_t i=0; i<bufExp.size(); ++i)
    {
        bufOut[i] = bufExp[i] - bufOut[i];
    }

    getLabelError(label);

    // learn
    net.bprop(&bufOut[0], NULL, 0.9);
}

void TrainMnist::Private::getLabelError(int label)
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
}

void TrainMnist::Private::testPerformance()
{
    clearErrorCount();

    net.layer(1)->setDropOut(MNN::DO_PERFORM);

    for (size_t num=0; num < testSet.numSamples(); ++num)
    {
        uint8_t label = testSet.label(num);
        const Float *image = testSet.image(num);

        net.fprop(image, &bufOut[0]);

        getLabelError(label);
    }

    net.layer(1)->setDropOut(MNN::DO_TRAIN);
    //net.layer(1)->setDropOut(MNN::DO_OFF);

    // output performance
    std::cout << "performance "
              << ", error " << error_min
              << " - " << error_max
              << ", pc";
    for (size_t j=0; j<errorsPerClass.size(); ++j)
        std::cout << " " << std::setw(3) << errorsPerClass[j];
    std::cout << ", E " << error_count
              << std::endl;
}

