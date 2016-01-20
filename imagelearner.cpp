/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/20/2016</p>
*/

#include <vector>
#include <sstream>

#include <QSize>

#include "imagelearner.h"
#include "image.h"
#include "mnn/layer.h"

struct ImageLearner::Private
{
    Private(ImageLearner* p)
        : p         (p)
        , net       (0)
        , image     (0)
        , sizeIn    (32, 32)
        , sizeOut   (16, 16)
        , learnRate (0.1)
        , errorAv   (0.)
        , epoch     (0)
    {

    }

    bool prepareTraining();
    void fpropPatch(int x, int y);
    void trainStep();
    void getExpectedPatch();
    void renderImageReconstruction(Image *dst);

    ImageLearner * p;

    MNN::Layer<Float>* net;
    Image* image, *imageRecon;

    QSize sizeIn, sizeOut;
    std::vector<Float>
        patchIn, patchOut,
        patchExpect, patchError;
    Float learnRate,
        errorAv;
    size_t epoch;
};


ImageLearner::ImageLearner()
    : p_        (new Private(this))
{

}

ImageLearner::~ImageLearner()
{
    if (p_->image)
        p_->image->releaseRef();
    if (p_->net)
        p_->net->releaseRef();
    delete p_;
}

const QSize& ImageLearner::sizeIn() const { return p_->sizeIn; }
const QSize& ImageLearner::sizeOut() const { return p_->sizeOut; }
const ImageLearner::Float* ImageLearner::patchIn() const { return &p_->patchIn[0]; }
const ImageLearner::Float* ImageLearner::patchOut() const { return &p_->patchOut[0]; }
const ImageLearner::Float* ImageLearner::patchExpect() const { return &p_->patchExpect[0]; }
const ImageLearner::Float* ImageLearner::patchError() const { return &p_->patchError[0]; }

void ImageLearner::setNet(MNN::Layer<Float> *net)
{
    if (p_->net)
        p_->net->releaseRef();
    p_->net = net;
    p_->net->addRef();

    p_->errorAv = 0.;
    p_->epoch = 0;
    /*
    // resize
    // XXX will break for conv nets
    int sin = p_->sizeIn.width() * p_->sizeIn.height(),
        sout = p_->sizeOut.width() * p_->sizeOut.height();
    if (p_->net->numIn() != sin || p_->net->numOut() != sout)
        p_->net->resize(sin, sout);
    */
}

void ImageLearner::setImage(Image *img)
{
    if (p_->image)
        p_->image->releaseRef();
    p_->image = img;
    p_->image->addRef();
}

void ImageLearner::trainStep(int iterations)
{
    if (p_->prepareTraining())
        for (int i=0; i<iterations; ++i)
            p_->trainStep();
}

bool ImageLearner::Private::prepareTraining()
{
    if (!net || !image)
        return false;

    if (sizeOut.width() > sizeIn.width()
        || sizeOut.height() > sizeIn.height())
        return false;

    if (sizeIn.width() >= (int)image->width()
      || sizeIn.height() >= (int)image->height())
        return false;

    patchIn.resize(sizeIn.width() * sizeIn.height());
    patchOut.resize(sizeOut.width() * sizeOut.height());
    patchExpect.resize(patchOut.size());
    patchError.resize(patchOut.size());

    if (net->numIn() != patchIn.size()
      || net->numOut() != patchOut.size())
        return false;

    return true;
}

void ImageLearner::fpropPatch(int x, int y)
{
    if (p_->prepareTraining())
    {
        x = std::max(0, std::min(int(p_->image->width() - p_->sizeIn.width()) - 1, x));
        y = std::max(0, std::min(int(p_->image->height() - p_->sizeIn.height()) - 1, y));
        p_->fpropPatch(x, y);
    }
}

void ImageLearner::Private::fpropPatch(int x, int y)
{
    image->getPatch(&patchIn[0], x, y, sizeIn.width(), sizeIn.height());

    // get output
    net->fprop(&patchIn[0], &patchOut[0]);

    getExpectedPatch();

    // get error
    for (size_t i=0; i<patchOut.size(); ++i)
        patchError[i] = patchExpect[i] - patchOut[i];
}

void ImageLearner::Private::trainStep()
{
    // get input patch
    int x = rand() % (image->width() - sizeIn.width());
    int y = rand() % (image->height() - sizeIn.height());

    fpropPatch(x, y);

    // get error
    Float error = 0.;
    for (size_t i=0; i<patchOut.size(); ++i)
        error += std::abs(patchError[i]);

    error = Float(100) * error / patchOut.size();
    errorAv += 1./100. * (error - errorAv);

    // train
    net->bprop(&patchError[0], 0, learnRate);

    ++epoch;
}

void ImageLearner::Private::getExpectedPatch()
{
    const int
            oy = (sizeIn.height() - sizeOut.height()) / 2,
            ox = (sizeIn.width() - sizeOut.width()) / 2;

    for (int j=0; j<sizeOut.height(); ++j)
    for (int i=0; i<sizeOut.width(); ++i)
    {
        // center outpatch in inpatch
        int iy = oy + j;
        int ix = ox + i;

        patchExpect[j * sizeOut.width() + i] =
                patchIn[iy * sizeIn.width() + ix];
    }
}

std::string ImageLearner::infoString() const
{
    std::stringstream s;
    s <<   "train step : " << p_->epoch
      << "\nav. error  : " << p_->errorAv
         ;
    return s.str();
}

void ImageLearner::renderImageReconstruction(Image *dst)
{
    p_->renderImageReconstruction(dst);
}

void ImageLearner::Private::renderImageReconstruction(Image *dst)
{
    if (!prepareTraining())
        return;

    dst->resize(image->width(), image->height(), 1);
    dst->clear();

    const int
            oy = (sizeIn.height() - sizeOut.height()) / 2,
            ox = (sizeIn.width() - sizeOut.width()) / 2;

    for (size_t j = 0; j < image->height() - sizeIn.height(); j += sizeOut.height())
    for (size_t i = 0; i < image->width() - sizeIn.width(); i += sizeOut.width())
    {
        // get input patch
        image->getPatch(&patchIn[0], i, j, sizeIn.width(), sizeIn.height());

        // get output patch
        net->fprop(&patchIn[0], &patchOut[0]);

        // store in dst
        dst->setPatch(&patchOut[0], ox+i, oy+j, sizeOut.width(), sizeOut.height());
    }
}
