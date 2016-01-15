/** @file mnistset.cpp

    @brief MNIST Handwritten Digits loader

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/21/2015</p>
*/

#include <cstdio>
#include <cassert>
#include <exception>
#include <iostream>

#include "mnistset.h"
#include "mnn/function.h"




MnistSet::MnistSet()
    : p_width_      (0)
    , p_height_     (0)
{

}

// swap byte order
void swap_bo(uint32_t& v)
{
    v =   ((v & 0xff) << 24)
        | (((v >> 8) & 0xff) << 16)
        | (((v >> 16) & 0xff) << 8)
        | (((v >> 24) & 0xff));
}

void MnistSet::load(const char* labelName, const char* imageName)
{
    // ----- read labels -----

    FILE* f = fopen(labelName, "rb");
    if (!f)
        throw mnist_exception("could not open label file");

    uint32_t v, num;
    fread(&v, 4, 1, f);
    swap_bo(v);
    if (v != 0x801)
    {
        fclose(f);
        throw mnist_exception("label file header not right");
    }

    fread(&num, 4, 1, f);
    swap_bo(num);
    p_labels_.resize(num);
    fread(&p_labels_[0], 1, num, f);

    fclose(f);

    // ----- read images -----

    f = fopen(imageName, "rb");
    if (!f)
        throw mnist_exception("could not open image file");

    fread(&v, 4, 1, f);
    swap_bo(v);
    if (v != 0x803)
    {
        fclose(f);
        throw mnist_exception("image file header not right");
    }

    fread(&num, 4, 1, f);
    swap_bo(num);

    if (num != p_labels_.size())
    {
        fclose(f);
        throw mnist_exception("wrong number of samples in image file");
    }

    fread(&p_width_, 4, 1, f);
    fread(&p_height_, 4, 1, f);
    swap_bo(p_width_);
    swap_bo(p_height_);

    p_images_.resize(p_width_ * p_height_ * num);
    std::vector<uint8_t> tmp(p_images_.size());

    if (tmp.size() != fread(&tmp[0], 1, tmp.size(), f))
    {
        fclose(f);
        throw mnist_exception("not enough data in image file");
    }

    fclose(f);

    for (size_t i=0; i<tmp.size(); ++i)
    {
        p_images_[i] = float(tmp[i]) / 255.f;
    }

}

float MnistSet::getMean() const
{
    float sum = 0.f;
    for (auto f : p_images_)
        sum += f;
    if (!p_images_.empty())
        sum /= p_images_.size();
    return sum;
}

void MnistSet::normalize()
{
    float mean = getMean();
    std::cout << "normalizing with mean value " << mean << std::endl;
    for (auto& f : p_images_)
        f -= mean;
}

const float* MnistSet::getNoisyBackgroundImage(
    uint32_t index, float backgroundThreshold, float minRnd, float maxRnd)
{
    if (p_processed_.size() != width() * height())
        p_processed_.resize(width() * height());

    const float * img = image(index);
    for (auto& p : p_processed_)
    {
        float pix = *img++;

        if (pix <= backgroundThreshold)
        {
            pix = MNN::rnd(minRnd, maxRnd);
        }

        p = pix;
    }

    return &p_processed_[0];
}

const float* MnistSet::getNoisyImage(
    uint32_t index, float minRnd, float maxRnd)
{
    if (p_processed_.size() != width() * height())
        p_processed_.resize(width() * height());

    const float * img = image(index);
    for (auto& p : p_processed_)
    {
        float pix = *img++;

        pix += MNN::rnd(minRnd, maxRnd);

        p = pix;
    }

    return &p_processed_[0];
}

uint32_t MnistSet::nextRandomSample(uint32_t index) const
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

void MnistSet::scale(uint32_t w, uint32_t h)
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
