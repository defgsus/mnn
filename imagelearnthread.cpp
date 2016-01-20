/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/20/2016</p>
*/

#include <QMutex>
#include <QMutexLocker>
#include <QTime>


#include "imagelearnthread.h"
#include "imagelearner.h"

struct ImageLearnThread::Private
{
    Private(ImageLearnThread * p)
        : p         (p)
        , learner   (new ImageLearner)
        , imageRecon(0)
    {

    }

    ~Private()
    {
        delete learner;
    }

    void mainLoop();

    ImageLearnThread* p;
    QMutex mutex;
    ImageLearner* learner;
    Image* imageRecon;
    volatile bool doStop;
    std::string infoStr;
};

ImageLearnThread::ImageLearnThread(QObject* parent)
    : QThread   (parent)
    , p_        (new Private(this))
{

}

ImageLearnThread::~ImageLearnThread()
{
    stop();
    delete p_;
}

void ImageLearnThread::stop()
{
    p_->doStop = true;
    wait();
}

std::string ImageLearnThread::infoString() const { return p_->infoStr; }
const ImageLearner* ImageLearnThread::imageLearner() const { return p_->learner; }
void ImageLearnThread::lock() { p_->mutex.lock(); }
void ImageLearnThread::unlock() { p_->mutex.unlock(); }


void ImageLearnThread::run() { p_->mainLoop(); }

void ImageLearnThread::setNet(MNN::Layer<Float> *net)
{
    QMutexLocker lock(&p_->mutex);
    p_->learner->setNet(net);
}

void ImageLearnThread::setImage(Image *img)
{
    QMutexLocker lock(&p_->mutex);
    p_->learner->setImage(img);
}

void ImageLearnThread::renderReconstruction(Image *img)
{
    p_->imageRecon = img;
}

void ImageLearnThread::fpropPatch(int x, int y)
{
    QMutexLocker lock(&p_->mutex);
    p_->learner->fpropPatch(x, y);
}

void ImageLearnThread::Private::mainLoop()
{
    doStop = false;
    QTime tm;
    tm.start();
    while (!doStop)
    {
        if (imageRecon)
        {
            learner->renderImageReconstruction(imageRecon);
            imageRecon = 0;
            emit p->reconstructionFinished();
        }

        {
            QMutexLocker lock(&mutex);
            learner->trainStep(20);
            infoStr = learner->infoString();
        }
        p->msleep(10);

        if (tm.elapsed() > 1000)
        {
            emit p->progress();
            tm.start();
        }
    }
}

