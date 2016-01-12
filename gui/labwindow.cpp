/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/25/2015</p>
*/

#include <QLayout>
#include <QPlainTextEdit>
#include <QMenu>
#include <QMenuBar>
#include <QTimer>
#include <QFileDialog>

#include "labwindow.h"
#include "statedisplay.h"
#include "mnn/mnn.h"
#include "trainthread.h"
#include "mnistset.h"



struct LabWindow::Private
{
    Private(LabWindow * win)
        : win       (win)
        , net       (1,1)
        , thread    (0)
    {
        //net = new MNN::Rbm<Float>(1, 1);
        //net->loadTextFile("../rbm_layer_0.txt");
        //net->loadTextFile("../mnist_rbm_100h.txt");
    }

    ~Private()
    {
        if (thread)
            thread->stop();
        delete thread;
        //delete net;
    }

    void createWidgets();
    void updateInfoWin(const MNN::Layer<Float>* net = 0);
    void updateStates();
    void saveLayer();
    // starts or pauses the thread
    void setTraining(bool enable);

    LabWindow * win;
    //MNN::Layer<Float> * net;
    TrainThread::NetType net;
    TrainThread* thread;
    QTimer* updateTimer;

    MnistSet mnist;

    QPlainTextEdit * infoWin, * stateWin;
    StateDisplay * stateDisplay, * stateDisplay2;
};



LabWindow::LabWindow(QWidget *parent)
    : QMainWindow   (parent)
    , p_            (new Private(this))
{
    setWindowTitle(tr("NN Lab"));
    setMinimumSize(480, 320);

    p_->createWidgets();

    p_->updateInfoWin();
    p_->updateStates();
}

LabWindow::~LabWindow()
{
    delete p_;
}


void LabWindow::Private::createWidgets()
{
    win->setCentralWidget(new QWidget(win));
    auto lv = new QVBoxLayout(win->centralWidget());

        infoWin = new QPlainTextEdit(win);
        lv->addWidget(infoWin);

        stateWin = new QPlainTextEdit(win);
        lv->addWidget(stateWin);

        stateDisplay = new StateDisplay(win);
        lv->addWidget(stateDisplay);

        stateDisplay2 = new StateDisplay(win);
        stateDisplay2->setZoom(16);
        lv->addWidget(stateDisplay2);

    // .. menu ..

    auto menu = win->menuBar()->addMenu(tr("main"));
    auto a = menu->addAction(tr("training"));
    a->setCheckable(true);
    connect(a, &QAction::triggered, [=](){ setTraining(a->isChecked()); });

    a = menu->addAction(tr("save layer"));
    connect(a, &QAction::triggered, [=](){ saveLayer(); });



    updateTimer = new QTimer(win);
    updateTimer->setSingleShot(false);
    updateTimer->setInterval(1000);
    connect(updateTimer, &QTimer::timeout, [=]()
    {
        win->onNetChanged();
        updateStates();
    });
}

void LabWindow::Private::updateInfoWin(const MNN::Layer<Float>* net)
{
    if (net == 0)
        net = &this->net;
    if (!net)
    {
        infoWin->clear();
        return;
    }
    std::stringstream s;
    net->info(s);
    infoWin->setPlainText(QString::fromStdString(s.str()));
}

void LabWindow::Private::updateStates()
{
    stateDisplay->setInstancesPerRow(20);
    stateDisplay->setStateSize(28, 28, net.numOut());
    stateDisplay->setStates(net.weights());

    stateDisplay2->setInstancesPerRow(20);
    stateDisplay2->setStateSize(1, 1, net.numOut());
    stateDisplay2->setStates(net.outputs());
}

void LabWindow::onNetChanged()
{
    p_->net = p_->thread->getNetCopy(0);
    p_->stateWin->setPlainText(QString::fromStdString(
                                  p_->thread->rbmStack().statusString(0)));
    p_->updateInfoWin();
}


void LabWindow::Private::setTraining(bool enable)
{
    if (!thread)
    {
        thread = new TrainThread(win);
        connect(thread, SIGNAL(rbmChanged()), win, SLOT(onNetChanged()));

        thread->setSize( { mnist.width() * mnist.height(), 4, 10 } );

        thread->start();
    }

    thread->setPause(!enable);
    if (enable)
        updateTimer->start();
    else
        updateTimer->stop();

}

void LabWindow::Private::saveLayer()
{
    QString fn = QFileDialog::getSaveFileName(win, tr("save layer"), "../");
    if (fn.isEmpty())
        return;

    auto net = thread->getNetCopy(0);
    net.saveTextFile(fn.toStdString());
}
