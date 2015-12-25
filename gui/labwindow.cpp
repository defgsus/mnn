/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/25/2015</p>
*/

#include <QLayout>
#include <QPlainTextEdit>

#include "labwindow.h"
#include "statedisplay.h"
#include "mnn/mnn.h"

typedef float Float;



struct LabWindow::Private
{
    Private(LabWindow * win) : win(win)
    {
        net = new MNN::Rbm<Float>(1, 1);
        net->loadTextFile("../rbm_layer_0.txt");
        //net->loadTextFile("../mnist_rbm_20h.txt");
    }

    ~Private()
    {
        delete net;
    }

    void createWidgets();
    void updateInfoWin(const MNN::Layer<Float>* net = 0);

    LabWindow * win;
    MNN::Layer<Float> * net;

    QPlainTextEdit * infoWin;
    StateDisplay * stateDisplay;
};



LabWindow::LabWindow(QWidget *parent)
    : QMainWindow   (parent)
    , p_            (new Private(this))
{
    setWindowTitle(tr("NN Lab"));
    setMinimumSize(480, 320);

    p_->createWidgets();

    p_->updateInfoWin();
    p_->stateDisplay->setStateSize(28, 28, 20);
    p_->stateDisplay->setStates(p_->net->weights());
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

        stateDisplay = new StateDisplay(win);
        lv->addWidget(stateDisplay);
}

void LabWindow::Private::updateInfoWin(const MNN::Layer<Float>* net)
{
    if (net == 0)
        net = this->net;
    std::stringstream s;
    net->info(s);
    infoWin->setPlainText(QString::fromStdString(s.str()));
}
