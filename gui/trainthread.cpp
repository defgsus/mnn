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

TrainThread::NetType TrainThread::getNetCopy(size_t index) const
{
    QReadLocker lock(mutex_);
    auto n = rbms.rbm(index);
    if (n)
        return *n;
    NetType net(1,1);
    return net;
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
        // update number cells
        if (!sizeRequest_.empty())
        {
            sizeRequest_[0] = mnist.width() * mnist.height();

            rbms.setSize(sizeRequest_);
            sizeRequest_.clear();

            // copy samples
            for (size_t i=0; i<mnist.numSamples(); ++i)
                rbms.addSample(mnist.image(i));

            emit rbmChanged();
        }

        if (doPause_ || rbms.samples().empty())
        {
            msleep(50);
            continue;
        }

        msleep(2);
        QWriteLocker lock(mutex_);

        for (int i=0; i<10; ++i)
            rbms.trainStep(trainLayerIndex_);
    }
}

