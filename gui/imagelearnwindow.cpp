/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/20/2016</p>
*/

#include <sstream>

#include <QLayout>
#include <QPlainTextEdit>
#include <QMenu>
#include <QMenuBar>
#include <QTimer>
#include <QFileDialog>
#include <QMessageBox>
#include <QTabWidget>
#include <QDoubleSpinBox>

#include "imagelearnwindow.h"
#include "statedisplay.h"
#include "image.h"
#include "imagelearner.h"
#include "imagelearnthread.h"
#include "mnn/exception.h"
#include "mnn/layer.h"
#include "mnn/factory.h"

struct ImageLearnWindow::Private
{
    Private(ImageLearnWindow * win)
        : win           (win)
        , net           (0)
        , thread        (new ImageLearnThread(win))
    {
        connect(thread, SIGNAL(progress()), win, SLOT(onProgress()),
                Qt::QueuedConnection);
        connect(thread, SIGNAL(reconstructionFinished()), win, SLOT(onReconstruction()),
                Qt::QueuedConnection);
    }

    void createWidgets();
    void updateInfo();
    void updateImageDisplay();
    void updateImageCorruptDisplay();
    void updateImageReconDisplay();
    void updatePatchDisplay();

    void setNetwork(MNN::Layer<Float>* net);
    void updateNetSettings();
    void createNet();

    void createCorruptedImage();

    ImageLearnWindow* win;

    Image image, imageCorrupt, imageRecon;
    MNN::Layer<Float>* net;
    ImageLearnThread * thread;
    bool stopAfterReconstruction;

    QPlainTextEdit * infoEdit;
    StateDisplay *imageDisplay, *imageReconDisplay, *imageCorruptDisplay,
        * patchInDisp, *patchOutDisp, *patchExpectDisp, *patchErrorDisp;
    QTabWidget *tabWidget;
    QDoubleSpinBox *spinLearnrate;
};

ImageLearnWindow::ImageLearnWindow(QWidget *parent)
    : QMainWindow   (parent)
    , p_            (new Private(this))
{
    setWindowTitle(tr("Image learner"));
    setMinimumSize(480, 320);
    setProperty("maximized", true);

    p_->createWidgets();

    p_->createNet();

    //loadImage("/home/defgsus/pic/weg_bochum.png");
    loadImage("/home/defgsus/prog/qt_project/nn/pics/font320.png");
}

ImageLearnWindow::~ImageLearnWindow()
{
    delete p_;
}


void ImageLearnWindow::Private::createWidgets()
{
    win->setCentralWidget(new QWidget(win));
    auto lv = new QVBoxLayout(win->centralWidget());

        auto lh = new QHBoxLayout();
        lv->addLayout(lh);

            auto lv1 = new QVBoxLayout();
            lh->addLayout(lv1);

                infoEdit = new QPlainTextEdit(win);
                lv1->addWidget(infoEdit, 2);

                spinLearnrate = new QDoubleSpinBox(win);
                spinLearnrate->setRange(0., 1.);
                spinLearnrate->setSingleStep(0.00001);
                spinLearnrate->setDecimals(9);
                spinLearnrate->setValue(0.0001);
                lv1->addWidget(spinLearnrate);
                connect(spinLearnrate, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged),
                        [=](){ updateNetSettings(); });

            tabWidget = new QTabWidget(win);
            lh->addWidget(tabWidget, 5);

                imageDisplay = new StateDisplay(win);
                tabWidget->addTab(imageDisplay, tr("original"));
                connect(imageDisplay, &StateDisplay::clicked, [=](int x, int y)
                {
                    thread->fpropPatch(x, y);
                    updatePatchDisplay();
                });

                imageCorruptDisplay = new StateDisplay(win);
                tabWidget->addTab(imageCorruptDisplay, tr("corrupted"));
                connect(imageCorruptDisplay, &StateDisplay::clicked, [=](int x, int y)
                {
                    thread->fpropPatch(x, y);
                    updatePatchDisplay();
                });

                imageReconDisplay = new StateDisplay(win);
                tabWidget->addTab(imageReconDisplay, tr("reconstruction"));
                connect(imageReconDisplay, &StateDisplay::clicked, [=](int x, int y)
                {
                    thread->fpropPatch(x, y);
                    updatePatchDisplay();
                });

            lv1 = new QVBoxLayout();
            lh->addLayout(lv1);

                patchInDisp = new StateDisplay(win);
                patchInDisp->setZoom(4);
                lv1->addWidget(patchInDisp);

                patchOutDisp = new StateDisplay(win);
                patchOutDisp->setZoom(4);
                lv1->addWidget(patchOutDisp);

                patchExpectDisp = new StateDisplay(win);
                patchExpectDisp->setZoom(4);
                lv1->addWidget(patchExpectDisp);

                patchErrorDisp = new StateDisplay(win);
                patchErrorDisp->setZoom(4);
                patchErrorDisp->setDisplayMode(StateDisplay::DM_SIGNED);
                lv1->addWidget(patchErrorDisp);

    win->setMenuBar(new QMenuBar(win));

    auto menu = win->menuBar()->addMenu(tr("Image"));

        auto a = menu->addAction(tr("Load image"));
        a->setShortcut(Qt::CTRL + Qt::SHIFT + Qt::Key_L);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(loadImage()));

        a = menu->addAction(tr("Load corrupted image"));
        connect(a, SIGNAL(triggered(bool)), win, SLOT(loadCorruptImage()));

    menu = win->menuBar()->addMenu(tr("Network"));

        a = menu->addAction(tr("New network"));
        a->setShortcut(Qt::CTRL + Qt::Key_N);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(newNetwork()));

        a = menu->addAction(tr("Load network"));
        a->setShortcut(Qt::CTRL + Qt::Key_L);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(loadNetwork()));

        a = menu->addAction(tr("Save network"));
        a->setShortcut(Qt::CTRL + Qt::Key_S);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(saveNetwork()));

        menu->addSeparator();

        a = menu->addAction(tr("Start training"));
        a->setShortcut(Qt::Key_F7);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(startThread()));

        a = menu->addAction(tr("Stop training"));
        a->setShortcut(Qt::Key_F8);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(stopThread()));

        menu->addSeparator();

        a = menu->addAction(tr("render reconstruction from corrupted"));
        a->setShortcut(Qt::Key_F9);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(renderImageReconstruction()));

        a = menu->addAction(tr("render reconstruction from org"));
        a->setShortcut(Qt::Key_F10);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(renderOrgImageReconstruction()));

}

void ImageLearnWindow::startThread()
{
    p_->stopAfterReconstruction = false;
    p_->thread->start();
}

void ImageLearnWindow::stopThread() { p_->thread->stop(); }

void ImageLearnWindow::loadImage()
{
    auto fn = QFileDialog::getOpenFileName(
                this, tr("Load image"),
                "", "images (*.jpg *.jpeg *.png *.bmp *.tif *.gif)");
    if (!fn.isEmpty())
        loadImage(fn);
}

void ImageLearnWindow::loadImage(const QString &fn)
{
    try
    {
        p_->image.loadFile(fn.toStdString());
        p_->createCorruptedImage();
        p_->thread->setImages(&p_->image, &p_->imageCorrupt);
    }
    catch (const MNN::Exception& e)
    {
        QMessageBox::critical(this, tr("Load image failed"),
                              e.what());
    }

    p_->updateImageDisplay();
}

void ImageLearnWindow::loadCorruptImage()
{
    auto fn = QFileDialog::getOpenFileName(
                this, tr("Load image"),
                "", "images (*.jpg *.jpeg *.png *.bmp *.tif *.gif)");
    if (!fn.isEmpty())
        loadCorruptImage(fn);
}

void ImageLearnWindow::loadCorruptImage(const QString &fn)
{
    try
    {
        p_->imageCorrupt.loadFile(fn.toStdString());
        p_->thread->setImages(&p_->image, &p_->imageCorrupt);
    }
    catch (const MNN::Exception& e)
    {
        QMessageBox::critical(this, tr("Load image failed"),
                              e.what());
    }

    p_->updateImageCorruptDisplay();
}

void ImageLearnWindow::newNetwork()
{
    p_->createNet();
}

void ImageLearnWindow::loadNetwork()
{
    auto fn = QFileDialog::getOpenFileName(
                this, tr("Load network"),
                "", "ascii nets (*)");
    if (!fn.isEmpty())
        loadNetwork(fn);
}

void ImageLearnWindow::loadNetwork(const QString &fn)
{
    try
    {
        auto net = MNN::Factory<Float>::loadTextFile(fn.toStdString());
        setWindowTitle("[" + fn + "]");
        p_->setNetwork(net);
    }
    catch (const MNN::Exception& e)
    {
        QMessageBox::critical(this, tr("Load network failed"),
                              e.what());
    }
}

void ImageLearnWindow::saveNetwork()
{
    auto fn = QFileDialog::getSaveFileName(
                this, tr("Save network"),
                "", "ascii nets (*)");
    if (!fn.isEmpty())
        saveNetwork(fn);
}

void ImageLearnWindow::saveNetwork(const QString &fn)
{
    try
    {
        p_->net->saveTextFile(fn.toStdString());
        setWindowTitle("[" + fn + "]");
    }
    catch (const MNN::Exception& e)
    {
        QMessageBox::critical(this, tr("Save network failed"),
                              e.what());
    }
}

void ImageLearnWindow::Private::updateInfo()
{
    if (!net)
        infoEdit->clear();
    else
    {
        std::stringstream s;
        net->info(s);
        infoEdit->setPlainText(QString::fromStdString(
            thread->infoString() + "\n\n" + s.str() ));
    }
}

void ImageLearnWindow::Private::updateImageDisplay()
{
    imageDisplay->setStateSize(image.width(), image.height());
    imageDisplay->setStates(image.data());
}

void ImageLearnWindow::Private::updateImageCorruptDisplay()
{
    imageCorruptDisplay->setStateSize(imageCorrupt.width(), imageCorrupt.height());
    imageCorruptDisplay->setStates(imageCorrupt.data());
}

void ImageLearnWindow::Private::updateImageReconDisplay()
{
    imageReconDisplay->setStateSize(imageRecon.width(), imageRecon.height());
    imageReconDisplay->setStates(imageRecon.data());
}

void ImageLearnWindow::Private::updatePatchDisplay()
{
    thread->lock();

    patchInDisp->setStateSize(thread->imageLearner()->sizeIn());
    patchInDisp->setStates(thread->imageLearner()->patchIn());
    patchOutDisp->setStateSize(thread->imageLearner()->sizeOut());
    patchOutDisp->setStates(thread->imageLearner()->patchOut());
    patchExpectDisp->setStateSize(thread->imageLearner()->sizeOut());
    patchExpectDisp->setStates(thread->imageLearner()->patchExpect());
    patchErrorDisp->setStateSize(thread->imageLearner()->sizeOut());
    patchErrorDisp->setStates(thread->imageLearner()->patchError());

    thread->unlock();
}

void ImageLearnWindow::Private::createNet()
{
    auto sizeIn = thread->imageLearner()->sizeIn(),
         sizeOut = thread->imageLearner()->sizeOut();

    auto stack = new MNN::StackSerial<Float>;

#if 0
    stack->addLayer( new MNN::FeedForward<Float, MNN::Activation::Linear>(
                                    sizeIn.width() * sizeIn.height(),
                                    sizeOut.width() * sizeOut.height()) );
#elif 0
    stack->addLayer( new MNN::FeedForward<Float, MNN::Activation::Tanh>(
                                    sizeIn.width() * sizeIn.height(),
                                    sizeOut.width() * sizeOut.height() * 8) );
    stack->addLayer( new MNN::FeedForward<Float, MNN::Activation::Linear>(
                                    stack->numOut(),
                                    sizeOut.width() * sizeOut.height()) );
#elif 0

    auto conv = new MNN::Convolution<Float, MNN::Activation::Tanh>(
                sizeIn.width(), sizeIn.height(), 1,
                16, 16, 8);
    stack->addLayer( conv );

    conv = new MNN::Convolution<Float, MNN::Activation::Tanh>(
                conv->scanWidth(), conv->scanHeight(), conv->numOutputMaps(),
                2, 2,
                10, 10, 1);
    stack->addLayer( conv );

    stack->addLayer( new MNN::FeedForward<Float, MNN::Activation::Linear>(
                                    stack->numOut(),
                                    sizeOut.width() * sizeOut.height())
                     );

#elif 1
    auto split = new MNN::StackSplit<Float>;

        split->addLayer( new MNN::FeedForward<Float, MNN::Activation::Tanh>
                                (sizeIn.width() * sizeIn.height(), 200) );

        auto convstack = new MNN::StackSerial<Float>;

            auto conv = new MNN::Convolution<Float, MNN::Activation::Tanh>(
                        sizeIn.width(), sizeIn.height(), 1,
                        16, 16, 8);
            convstack->addLayer( conv );

            conv = new MNN::Convolution<Float, MNN::Activation::Tanh>(
                        conv->scanWidth(), conv->scanHeight(), conv->numOutputMaps(),
                        2, 2,
                        10, 10, 1);
            convstack->addLayer( conv );

        split->addLayer(convstack);

    stack->addLayer(split);
    stack->addLayer( new MNN::FeedForward<Float, MNN::Activation::Linear>(
                                    stack->numOut(),
                                    sizeOut.width() * sizeOut.height())
                     );

#endif

    stack->brainwash(0.1);

    setNetwork(stack);
}

void ImageLearnWindow::Private::updateNetSettings()
{
    MNN::setLearnRate(net, float(spinLearnrate->value()));
    MNN::setLearnRateBias(net, float(spinLearnrate->value()) );
    MNN::setMomentum(net, 0.5f);
}

void ImageLearnWindow::Private::setNetwork(MNN::Layer<Float>* n)
{

    thread->setNet(n);

    thread->lock();
    if (net)
        net->releaseRef();
    net = n;
    updateNetSettings();
    thread->unlock();

    updateInfo();
}

void ImageLearnWindow::onProgress()
{
    p_->updateInfo();
    p_->updatePatchDisplay();
}

void ImageLearnWindow::onReconstruction()
{
    if (p_->stopAfterReconstruction)
    {
        p_->thread->stop();
        p_->stopAfterReconstruction = false;
    }
    p_->updateImageReconDisplay();
}

void ImageLearnWindow::renderImageReconstruction()
{
    p_->thread->renderReconstruction(&p_->imageRecon);
    if (!p_->thread->isRunning())
    {
        p_->stopAfterReconstruction = true;
        p_->thread->start();
    }
}

void ImageLearnWindow::renderOrgImageReconstruction()
{
    p_->thread->renderReconstruction(&p_->image, &p_->imageRecon);
    if (!p_->thread->isRunning())
    {
        p_->stopAfterReconstruction = true;
        p_->thread->start();
    }
}

void ImageLearnWindow::Private::createCorruptedImage()
{
#if 1
    imageCorrupt.resize(image.width(), image.height(), 1);
    for (size_t j=0; j<image.height(); ++j)
    for (size_t i=0; i<image.width(); ++i)
    {
#if 0
        // no corruption
        imageCorrupt.data()[j * image.width() + i]
                = image.data()[j * image.width() + i];
#elif 0
        // noisy
        imageCorrupt.data()[j * image.width() + i]
                = std::max(0.f, std::min(1.f,
                    image.data()[j * image.width() + i]
                    + MNN::rnd(-.3f, .3f) ));
#elif 1
        // probability dropout
        imageCorrupt.data()[j * image.width() + i]
                = MNN::rnd(0., 1.) < .75 ? 1.f : image.data()[j * image.width() + i];
#elif 1
        // resolution reduction
        int qi = int(i / 3) * 3;
        int qj = int(j / 3) * 3;
        imageCorrupt.data()[j * image.width() + i]
                = image.data()[qj * image.width() + qi];
#endif
    }
#else
    imageCorrupt.resize(image.width()/3, image.height()/3, 1);
    for (size_t j=0; j<imageCorrupt.height(); ++j)
    for (size_t i=0; i<imageCorrupt.width(); ++i)
        imageCorrupt.data()[j * imageCorrupt.width() + i]
                = image.data()[j*3 * image.width() + i*3];
#endif

    updateImageCorruptDisplay();
}
