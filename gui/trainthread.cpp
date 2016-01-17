/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/25/2015</p>
*/

#include <QReadWriteLock>
#include <QReadLocker>
#include <QWriteLocker>

#include "trainthread.h"
#include "mnistset.h"

TrainThread::TrainThread(QObject *parent)
    : QThread           (parent)
    , trainLayerIndex_  (0)
    , mutex_            (new QReadWriteLock)
{

}

TrainThread::~TrainThread()
{
    delete mutex_;
}

TrainThread::NetType* TrainThread::getNetCopy(size_t index) const
{
    QReadLocker lock(mutex_);
    auto n = rbms_.rbm(index);
    if (n)
        return dynamic_cast<NetType*>(n->getCopy());
    return 0;
}


void TrainThread::run()
{
    doRun_ = true;
    doPause_ = false;

    MnistSet mnist;
    mnist.load("/home/defgsus/prog/DATA/mnist/t10k-labels.idx1-ubyte",
               "/home/defgsus/prog/DATA/mnist/t10k-images.idx3-ubyte");

    while (doRun_)
    {
        msleep(2);
        QWriteLocker lock(mutex_);

        // update number cells
        if (!sizeRequest_.empty())
        {
            sizeRequest_[0] = mnist.width() * mnist.height();

            rbms_.setSize(sizeRequest_);
            sizeRequest_.clear();

            // copy samples
            for (size_t i=0; i<mnist.numSamples(); ++i)
                rbms_.addSample(mnist.image(i));

            emit rbmChanged();
        }

        if (doPause_ || rbms_.samples().empty())
        {
            msleep(50);
            continue;
        }

        // -- training step --

        for (int i=0; i<16; ++i)
            rbms_.trainStep(trainLayerIndex_);
    }
}

