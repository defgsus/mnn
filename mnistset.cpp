/** @file mnistset.cpp

    @brief MNIST Handwritten Digits loader

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/21/2015</p>
*/

#include <cstdio>
#include <exception>

#include "mnistset.h"





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
        p_images_[i] = float(tmp[i]) / 255;
    }

}

