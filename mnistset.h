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

    void scale(uint32_t width, uint32_t height);

    // ----------- getter -----------

    uint32_t width() const { return p_width_; }
    uint32_t height() const { return p_height_; }
    uint32_t numSamples() const { return p_labels_.size(); }
    uint32_t numClasses() const { return 10; }

          uint8_t label(uint32_t index) const { return p_labels_[index]; }
    const float*  image(uint32_t index) const {
                                    return &p_images_[index * width() * height()]; }

    /** Returns the next random sample number with a different label */
    uint32_t nextRandomSample(uint32_t index) const;

    /** Returns the given image with a noisy background.
        Background is considered as values below @p backgroundThreshold.
        @note Returned pointer is valid until next call to this function. */
    const float* getNoisyBackgroundImage(
            uint32_t index, float backgroundThreshold, float minRnd, float maxRnd);
    /** @note Returned pointer is valid until next call to this function. */
    const float* getNoisyImage(
            uint32_t index, float minRnd, float maxRnd);

    const float* getTransformedImage(uint32_t index, float maxMorphPixels);

    // ------------- io -------------

    /** Throws MNN::Exception on any error */
    void load(const char* labelName, const char* imageName);

    /** Returns the mean of all pixel values */
    float getMean() const;

    /** Sets the mean for all pixel values to zero */
    void normalize();

private:

    uint32_t p_width_, p_height_;
    std::vector<float> p_images_, p_processed_;
    std::vector<uint8_t> p_labels_;
};


#endif // MNISTSET_H_INCLUDED
