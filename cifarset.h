/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/15/2016</p>
*/

#ifndef MNN_CIFARSET_H
#define MNN_CIFARSET_H

#include <cinttypes>
#include <vector>

/** Currently grayscale CIFAR-10 support */
class CifarSet
{
public:
    CifarSet();

    void scale(uint32_t width, uint32_t height);

    // ----------- getter -----------

    uint32_t width() const { return p_width_; }
    uint32_t height() const { return p_height_; }
    uint32_t numSamples() const { return p_labels_.size(); }
    uint32_t numClasses() const { return 10; }

          uint8_t label(uint32_t index) const { return p_labels_[index]; }
    const float*  image(uint32_t index) const {
                                    return &p_images_[index * width() * height()]; }
    /** @note Returned pointer is valid until next call to this function. */
    const float* getNoisyImage(
            uint32_t index, float minRnd, float maxRnd);

    /** Returns the next random sample number with a different label */
    uint32_t nextRandomSample(uint32_t index) const;

    // ------------- io -------------

    /** Clears all data */
    void clear();

    /** Throws MNN::Exception on any error */
    void load(const char* filename);

    /** Returns the mean of all pixel values */
    float getMean() const;

    /** Sets the mean for all pixel values to zero */
    void normalize();

private:

    uint32_t p_width_, p_height_;
    std::vector<float> p_images_, p_processed_;
    std::vector<uint8_t> p_labels_;
};

#endif // MNN_CIFARSET_H
