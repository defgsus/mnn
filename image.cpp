/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/20/2016</p>
*/

#include <QImageReader>
#include <QImage>

#include "image.h"
#include "mnn/exception.h"

Image::Image()
    : w_        (0)
    , h_        (0)
    , chan_     (0)
{

}


void Image::resize(size_t w, size_t h, size_t numChan)
{
    w_ = w;
    h_ = h;
    chan_ = numChan;
    data_.resize(w_ * h_ * chan_);
}

void Image::clear(float val)
{
    for (auto& v : data_)
        v = val;
}

void Image::loadFile(const std::string &fn)
{
    QImageReader r(QString::fromStdString(fn));
    QImage img = r.read();
    if (img.isNull())
        MNN_EXCEPTION("Can't open image file '"
                      << fn << "'\n" << r.errorString().toStdString());

    set(img);
}

void Image::set(const QImage& constImg)
{
    QImage tmp;
    const QImage* img = &constImg;
    // XXX Currently only grayscale support
    if (!img->format() == QImage::Format_Grayscale8)
    {
        tmp = img->convertToFormat(QImage::Format_Grayscale8);
        img = &tmp;
    }

    resize(img->width(), img->height(), 1);
    for (size_t j=0; j<h_; ++j)
    for (size_t i=0; i<w_; ++i)
        data_[j * w_ + i] = float(qRed(img->pixel(i, j))) / 255;
}

void Image::getPatch(float *data, int x, int y, size_t w, size_t h) const
{
    if (x >= (int)w_ || y >= (int)h_
       || x + (int)w < 0 || y + (int)h < 0)
    {
        for (size_t i=0; i<w*h; ++i)
            *data++ = 0.f;
    }

    size_t hclip = std::min(h, size_t(h_ - y - 1)),
           wclip = std::min(w, size_t(w_ - x - 1));

    for (size_t j = 0; j < hclip; ++j)
    for (size_t i = 0; i < wclip; ++i)
    {
        data[j * w + i] = data_[(y+j)*w_ + x+i];
    }

    for (size_t j = hclip; j < h; ++j)
    for (size_t i = wclip; i < w; ++i)
        data[j * w + i] = 0.f;
}


void Image::setPatch(const float *data, int x, int y, size_t w, size_t h)
{
    if (x >= (int)w_ || y >= (int)h_
       || x + (int)w < 0 || y + (int)h < 0)
        return;

    size_t hclip = std::min(h, size_t(h_ - y - 1)),
           wclip = std::min(w, size_t(w_ - x - 1));

    for (size_t j = 0; j < hclip; ++j)
    for (size_t i = 0; i < wclip; ++i)
    {
        data_[(y+j)*w_ + x+i] = data[j * w + i];
    }
}
