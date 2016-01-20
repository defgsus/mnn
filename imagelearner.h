/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/20/2016</p>
*/

#ifndef IMAGELEARNER_H
#define IMAGELEARNER_H

#include <string>

#include <QSize>

namespace MNN { template <typename Float> class Layer; }
class Image;

class ImageLearner
{
public:
    typedef float Float;

    ImageLearner();
    ~ImageLearner();

    const QSize& sizeIn() const;
    const QSize& sizeOut() const;

    const Float* patchIn() const;
    const Float* patchOut() const;
    const Float* patchExpect() const;
    const Float* patchError() const;

    /** Sets the network to use. Adds reference */
    void setNet(MNN::Layer<Float>* net);
    /** Sets the image to learn. Adds reference */
    void setImage(Image* img);

    void trainStep(int iterations);

    /** Selects an input patch from the current image
        and propagtes through network.
        patchIn(), patchOut(), patchExpect() and patchError()
        are all updated.
        @p x and @p y are clamped to image boundaries. */
    void fpropPatch(int x, int y);

    /** Rerender the current image through the network into dst.
        The image is resized appropriately. */
    void renderImageReconstruction(Image* dst);

    std::string infoString() const;

private:
    struct Private;
    Private * p_;
};

#endif // IMAGELEARNER_H
