/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/20/2016</p>
*/

#ifndef IMAGE_H
#define IMAGE_H

#include <cstddef>
#include <vector>
#include <string>

#include "mnn/refcounted.h"

class QImage;

class Image : public MNN::RefCounted
{
public:
    Image();

    size_t width() const { return w_; }
    size_t height() const { return h_; }
    const float* data() const { return data_.empty() ? 0 : &data_[0]; }
    float* data() { return data_.empty() ? 0 : &data_[0]; }

    void resize(size_t w, size_t h, size_t numChan);
    void clear(float val = 0.f);

    void loadFile(const std::string& fn);

    void set(const QImage& img);

    void getPatch(float* data, int x, int y, size_t w, size_t h) const;
    void setPatch(const float* data, int x, int y, size_t w, size_t h);

private:

    std::vector<float> data_;
    size_t w_, h_, chan_;

};

#endif // IMAGE_H
