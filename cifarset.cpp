/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/15/2016</p>
*/

#include <cstdio>
#include <cassert>
#include <exception>
#include <iostream>

#include "cifarset.h"
#include "mnn/function.h"
#include "mnn/exception.h"

CifarSet::CifarSet()
    : p_width_      (0)
    , p_height_     (0)
{

}


void CifarSet::load(const char* filename)
{
    // ----- read labels -----

    FILE* f = fopen(filename, "rb");
    if (!f)
        MNN_EXCEPTION("Could not open CIFAR file '" << filename << "'");

    p_width_ = p_height_ = 32;
    const size_t isize = p_width_ * p_height_;
    std::vector<uint8_t> rgb(isize * 3);

    for (int i=0; i<10000; ++i)
    {
        uint8_t label;
        fread(&label, 1, 1, f);
        p_labels_.push_back(label);

        fread(&rgb[0], 1, isize * 3, f);

        for (size_t j = 0; j < isize; ++j)
        {
            float red = float(rgb[j]) / 255,
                  green = float(rgb[j+isize]) / 255,
                  blue = float(rgb[j+isize*2]) / 255;
            p_images_.push_back(
                          .27 * red
                        + .60 * green
                        + .13 * blue);
        }
    }

    fclose(f);
}

float CifarSet::getMean() const
{
    float sum = 0.f;
    for (auto f : p_images_)
        sum += f;
    if (!p_images_.empty())
        sum /= p_images_.size();
    return sum;
}

void CifarSet::normalize()
{
    float mean = getMean();
    std::cout << "normalizing with mean value " << mean << std::endl;
    for (auto& f : p_images_)
        f -= mean;
}


uint32_t CifarSet::nextRandomSample(uint32_t index) const
{
    auto lab = label(index);

    size_t count = 0;
    do
    {
        index = rand() % numSamples();
    }
    while (lab == label(index) && count++ < numSamples());

    return index;
}

void CifarSet::scale(uint32_t w, uint32_t h)
{
    assert(w < width() && h < height());

    std::vector<float> scaled;
    for (uint32_t i = 0; i < numSamples(); ++i)
    {
        const float* img = image(i);
        for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x)
        {
            scaled.push_back(
                        img[(y * height() / h) * width() + x * width() / w]
                    );
        }
    }
    p_width_ = w;
    p_height_ = h;
    p_images_ = scaled;
}
