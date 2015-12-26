/** @file mnistset.h

    @brief MNIST Handwritten Digits loader

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/21/2015</p>
*/

#ifndef MNISTSET_H_INCLUDED
#define MNISTSET_H_INCLUDED

#include <cinttypes>
#include <vector>

class MnistSet
{
public:
    MnistSet();

    // ----------- getter -----------

    uint32_t width() const { return p_width_; }
    uint32_t height() const { return p_height_; }
    uint32_t numSamples() const { return p_labels_.size(); }

    uint8_t label(uint32_t index) const { return p_labels_[index]; }
    const float* image(uint32_t index) const
        { return &p_images_[index * width() * height()]; }

    // ------------- io -------------

    /** Throws mnist_exception on any error */
    void load(const char* labelName, const char* imageName);

private:

    uint32_t p_width_, p_height_;
    std::vector<float> p_images_;
    std::vector<uint8_t> p_labels_;
};


class mnist_exception : public std::exception
{
    const char * what_;
public:
  mnist_exception(const char* what) noexcept : what_(what) { }
  virtual ~mnist_exception() noexcept { }

  virtual const char* what() const noexcept { return what_; }
};

#endif // MNISTSET_H_INCLUDED
