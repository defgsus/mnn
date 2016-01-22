/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/20/2016</p>
*/

#ifndef IMAGELEARNTHREAD_H
#define IMAGELEARNTHREAD_H

#include <string>

#include <QThread>

namespace MNN { template <typename Float> class Layer; }
class Image;
class ImageLearner;

class ImageLearnThread
        : public QThread
{
    Q_OBJECT
public:
    typedef float Float;

    ImageLearnThread(QObject* parent);
    ~ImageLearnThread();

    const ImageLearner* imageLearner() const;

    void lock();
    void unlock();

    std::string infoString() const;

    /** Sets the network to use. Adds reference */
    void setNet(MNN::Layer<Float>* net);
    /** Sets the image to learn. Adds reference */
    void setImage(Image* img);
    /** Sets the corrupted image to learn. Adds reference */
    void setImageCorrupt(Image* img);
    /** Sets both at once */
    void setImages(Image* goal, Image* corrupted);

    /** Selects an input patch from the current image
        and propagtes through network.
        patchIn(), patchOut(), patchExpect() and patchError()
        are all updated.
        @p x and @p y are clamped to image boundaries.
        Executed immidiately and threadsafe. */
    void fpropPatch(int x, int y);

    /** Adds a request to reconstruct the current image
        through the network into @p dst.
        The image is resized appropriately.
        reconstructionFinished() is emitted when done. */
    void renderReconstruction(Image* dst);

    /** Adds a request to pass image @p src patch by patch
        into image @p dst. @p dst is resized appropriately.
        reconstructionFinished() is emitted when done. */
    void renderReconstruction(Image* src, Image* dst);

public slots:

    /** Blocks until end of thread. */
    void stop();

signals:

    /** Emitted regularily */
    void progress();

    /** Emitted when requested reconstruction is finished */
    void reconstructionFinished();

protected:

    void run() override;

private:
    struct Private;
    Private* p_;

};

#endif // IMAGELEARNTHREAD_H
