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
    void updateImageReconDisplay();
    void updatePatchDisplay();

    void setNetwork(MNN::Layer<Float>* net);
    void createNet();

    ImageLearnWindow* win;

    Image image, imageRecon;
    MNN::Layer<Float>* net;
    ImageLearnThread * thread;
    bool stopAfterReconstruction;

    QPlainTextEdit * infoEdit;
    StateDisplay *imageDisplay, *imageReconDisplay,
        * patchInDisp, *patchOutDisp, *patchExpectDisp, *patchErrorDisp;
    QTabWidget *tabWidget;
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

    loadImage("/home/defgsus/pic/weg_bochum.png");
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

            infoEdit = new QPlainTextEdit(win);
            lh->addWidget(infoEdit, 2);

            tabWidget = new QTabWidget(win);
            lh->addWidget(tabWidget, 5);

                imageDisplay = new StateDisplay(win);
                tabWidget->addTab(imageDisplay, tr("original"));
                connect(imageDisplay, &StateDisplay::clicked, [=](int x, int y)
                {
                    thread->fpropPatch(x, y);
                    updatePatchDisplay();
                });

                imageReconDisplay = new StateDisplay(win);
                tabWidget->addTab(imageReconDisplay, tr("reconstruction"));

            auto lv1 = new QVBoxLayout();
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

    auto menu = win->menuBar()->addMenu(tr("File"));

        auto a = menu->addAction(tr("Load image"));
        a->setShortcut(Qt::CTRL + Qt::SHIFT + Qt::Key_L);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(loadImage()));

        menu->addSeparator();

        a = menu->addAction(tr("Load network"));
        a->setShortcut(Qt::CTRL + Qt::Key_L);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(loadNetwork()));

        a = menu->addAction(tr("Save network"));
        a->setShortcut(Qt::CTRL + Qt::Key_S);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(saveNetwork()));

    menu = win->menuBar()->addMenu(tr("Network"));

        a = menu->addAction(tr("Start training"));
        a->setShortcut(Qt::Key_F7);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(startThread()));

        a = menu->addAction(tr("Stop training"));
        a->setShortcut(Qt::Key_F8);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(stopThread()));

        menu->addSeparator();

        a = menu->addAction(tr("render reconstruction"));
        a->setShortcut(Qt::Key_F9);
        connect(a, SIGNAL(triggered(bool)), win, SLOT(renderImageReconstruction()));

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
        p_->thread->setImage(&p_->image);
    }
    catch (const MNN::Exception& e)
    {
        QMessageBox::critical(this, tr("Load image failed"),
                              e.what());
    }

    p_->updateImageDisplay();
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
        p_->setNetwork(net);
        setWindowTitle(fn);
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

    stack->addLayer( new MNN::FeedForward<Float, MNN::Activation::Linear>(
                                    sizeIn.width() * sizeIn.height(),
                                    sizeOut.width() * sizeOut.height()) );

    stack->brainwash(0.1);
    stack->setLearnRate(0.01);

    setNetwork(stack);
}

void ImageLearnWindow::Private::setNetwork(MNN::Layer<Float>* n)
{
    thread->setNet(n);

    thread->lock();
    if (net)
        net->releaseRef();
    net = n;
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
