/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/15/2016</p>
*/

#include <QLayout>
#include <QAction>
#include <QMenu>
#include <QMenuBar>
#include <QFileDialog>
#include <QMessageBox>

#include "analyzewindow.h"
#include "statedisplay.h"
#include "mnn/mnn.h"


struct AnalyzeWindow::Private
{
    typedef MNN::FeedForward<float, MNN::Activation::Linear> NNLayer;

    Private(AnalyzeWindow * win )
        : win       (win)
        , layer     (0)
    {

    }

    void createWidgets();
    void updateStateDisp();
    void openLayer(const QString& fn);

    AnalyzeWindow * win;

    StateDisplay *stateDisp, *biasDisp;

    NNLayer* layer;
};


AnalyzeWindow::AnalyzeWindow(QWidget* parent, Qt::WindowFlags flags)
    : QMainWindow(parent, flags)
    , p_        (new Private(this))
{
    p_->createWidgets();
}

AnalyzeWindow::~AnalyzeWindow()
{
    delete p_;
}


void AnalyzeWindow::Private::createWidgets()
{
    win->setCentralWidget(new QWidget(win));
    auto lv = new QVBoxLayout(win->centralWidget());

        biasDisp = new StateDisplay(win);
        biasDisp->setZoom(16);
        lv->addWidget(biasDisp, 2);

        stateDisp = new StateDisplay(win);
        lv->addWidget(stateDisp, 5);

    win->setMenuBar(new QMenuBar(win));

    auto a = win->menuBar()->addAction(tr("open layer"));
    connect(a, SIGNAL(triggered(bool)), win, SLOT(openLayer()));
}

void AnalyzeWindow::openLayer()
{
    auto fn = QFileDialog::getOpenFileName(this, tr("load nn layer txt"), "../");
    if (fn.isEmpty())
        return;

    p_->openLayer(fn);

    p_->updateStateDisp();
}

void AnalyzeWindow::Private::openLayer(const QString &fn)
{
    try
    {
        if (!layer)
            layer = new Private::NNLayer(1, 1);

        layer->loadTextFile(fn.toStdString());
    }
    catch (const MNN::Exception& e)
    {
        QMessageBox::critical(win, tr("load nn layer failed"), e.what());
        delete layer;
        layer = 0;
    }

    updateStateDisp();
}

void AnalyzeWindow::Private::updateStateDisp()
{
    if (!layer)
       return;

    stateDisp->setInstancesPerRow(16);
    int s = std::sqrt(layer->numIn());
    stateDisp->setStateSize(s, s, layer->numOut());
    stateDisp->setStates(layer->weights());

    biasDisp->setStateSize(s, s, 1);
    biasDisp->setStates(layer->biases());
}
