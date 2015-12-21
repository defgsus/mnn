/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/21/2015</p>
*/

#include "trainmnist.h"
#include "mnn/mnn.h"
#include "mnistset.h"

struct TrainMnist::Private
{
    void exec();
    void trainStep();

    typedef float Float;

    MnistSet trainSet, testSet;
    MNN::StackSerial<Float> net;
    std::vector<Float> bufExp, bufOut;
    Float error, avError;
};

TrainMnist::TrainMnist()
    : p_        (new Private())
{

}

TrainMnist::~TrainMnist()
{
    delete p_;
}

void TrainMnist::exec() { p_->exec(); }

void TrainMnist::Private::exec()
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

    size_t numIn = trainSet.width() * trainSet.height();
    bufExp.resize(10);
    bufOut.resize(10);

    net.add( new MNN::Perceptron<Float, MNN::Activation::Logistic>(numIn, 800) );
    net.add( new MNN::Perceptron<Float, MNN::Activation::Logistic>(800, 10) );

    net.brainwash();

    net.info();

    // --------- training ---------

    avError = 0.;

    for (int i=0; i<1000000; ++i)
    {
        trainStep();

        if (i % 100 == 0)
        {
            std::cout << "epoch " << i
                      << ", error " << error
                      << std::endl;
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

    net.bprop(&bufOut[0], NULL, 0.9);
}

