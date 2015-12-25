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
    void createNet();
    void train();
    void trainStep();

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

void TrainMnist::Private::createNet()
{
    size_t numIn = trainSet.width() * trainSet.height();
    bufExp.resize(10);
    bufOut.resize(10);

    auto l1 = new MNN::Rbm<Float, MNN::Activation::Logistic>(numIn, 100);
    auto l2 = new MNN::Rbm<Float, MNN::Activation::Logistic>(100, 10);

    net.brainwash();

    // load previous
#if 1
    l1->loadTextFile("mnist_rbm_100h.txt");
#else
    l1->loadTextFile("rbm_layer_0.txt");
    l2->loadTextFile("rbm_layer_1.txt");
#endif

    l1->setMomentum(.1);
    l2->setMomentum(.9);
    net.add( l1 );
    net.add( l2 );
}

void TrainMnist::Private::train()
{
    // --------- training ---------

    error_count = 0;
    error_sum = error_max = 0.;
    error_min = -1.;
    errorsPerClass.resize(10);
    for (auto&x : errorsPerClass)
            x = 0.;

    for (int i=0; i<1000000; ++i)
    {
        trainStep();

        const int num = 5000;
        if (i % num == 0)
        {
            std::cout << "epoch " << i
                      << ", error " << error
                      << ", min " << error_min
                      << ", max " << error_max
                      << ", pc";
            for (size_t j=0; j<errorsPerClass.size(); ++j)
                std::cout << " " << std::setw(3) << errorsPerClass[j];
            std::cout << ", % " << (float(error_count) / num * 100)
                      << std::endl;

            error_count = 0;
            error_max = 0.;
            error_min = -1.;
            for (auto&x : errorsPerClass)
                    x = 0.;
        }
    }
}

void TrainMnist::Private::trainStep()
{
    // choose one
    size_t num = size_t(rand()) % trainSet.numSamples();
    uint8_t label = trainSet.label(num);
    const Float *image = trainSet.image(num);

    // prepare expected output
    for (auto& f : bufExp)
        f = 0.1;
    bufExp[label] = .9;

    net.fprop(image, &bufOut[0]);
    //std::cout << "---\n"; net.dump();

    // calc error
    error = 0.;
    for (size_t i=0; i<bufExp.size(); ++i)
    {
        //std::cout << bufOut[i] << " ";
        bufOut[i] = bufExp[i] - bufOut[i];
        error += std::abs(bufOut[i]);
    }
    //std::cout << std::endl;

#if 1
    // get error from actual label number
    int answer = -1;
    Float ma = -100.;
    for (size_t i=0; i<net.numOut(); ++i)
    {
        Float o = net.output(i);
        if (o > ma)
        {
            ma = o;
            answer = i;
        }
    }

    error = answer < 0 ? 11. : std::abs(Float(answer - label));
    if (answer != label)
    {
        ++error_count;
        ++errorsPerClass[label];
    }
#endif

    // get error stats
    error_max = std::max(error_max, error);
    if (error_min < 0.)
        error_min = error;
    else
        error_min = std::min(error_min, error);
    error_sum += error;

    // learn
    net.bprop(&bufOut[0], NULL, 0.9);
}

