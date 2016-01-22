/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/20/2016</p>
*/

#include <sstream>

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
        , imageReconSrc(0)
        , fps       (0)
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
    Image* imageRecon, *imageReconSrc;
    volatile bool doStop;
    std::string infoStr;
    float fps;
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

void ImageLearnThread::setImageCorrupt(Image *img)
{
    QMutexLocker lock(&p_->mutex);
    p_->learner->setCorruptedImage(img);
}

void ImageLearnThread::setImages(Image *img, Image* cor)
{
    QMutexLocker lock(&p_->mutex);
    p_->learner->setImage(img);
    p_->learner->setCorruptedImage(cor);
}

void ImageLearnThread::renderReconstruction(Image *img)
{
    p_->imageRecon = img;
}

void ImageLearnThread::renderReconstruction(Image* src, Image *dst)
{
    p_->imageReconSrc = src;
    p_->imageRecon = dst;
}

void ImageLearnThread::fpropPatch(int x, int y)
{
    QMutexLocker lock(&p_->mutex);
    p_->learner->fpropPatch(x, y);
}

void ImageLearnThread::Private::mainLoop()
{
    doStop = false;
    size_t lastEpoch = learner->epoch();
    QTime tm;
    tm.start();
    while (!doStop)
    {
        // render reconstruction
        if (imageRecon)
        {
            if (imageReconSrc)
                ImageLearner::renderImageReconstruction(
                            learner->net(),
                            learner->sizeIn(),
                            learner->sizeOut(),
                            imageReconSrc, imageRecon);
            else
                learner->renderImageReconstruction(imageRecon);
            imageRecon = 0;
            imageReconSrc = 0;
            emit p->reconstructionFinished();
        }

        // training
        {
            QMutexLocker lock(&mutex);
            learner->trainStep(20);
        }
        p->msleep(10);

        float elapsed = float(tm.elapsed()) / 1000.;
        if (elapsed > 1.f)
        {
            fps = (learner->epoch() - lastEpoch) / elapsed;
            lastEpoch = learner->epoch();

            std::stringstream s;
            s << learner->infoString()
              << "\nfps             : " << fps
              << "\nfull image time : "
                << (learner->stepsPerImage() / fps / 60) << " min";

            infoStr = s.str();

            emit p->progress();
            tm.start();
        }
    }
}

