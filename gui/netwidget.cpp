/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/21/2016</p>
*/

#include <QLayout>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QComboBox>

#include "netwidget.h"
#include "mnn/mnn.h"

struct NetWidget::Private
{
    struct Layer
    {
        QComboBox
            *comboType,
            *comboAct;
    };

    Private(NetWidget* p)
        : p             (p)
    { }

    Layer createLayerStruct();

    NetWidget* p;

    QList<Layer> layer;
};


NetWidget::NetWidget(QWidget *parent)
    : QWidget   (parent)
    , p_        (new Private(this))
{

}

NetWidget::~NetWidget()
{
    delete p_;
}


NetWidget::Private::Layer NetWidget::Private::createLayerStruct()
{
    Layer l;
    l.comboType = new QComboBox(p);
    l.comboType->addItem(tr("feed-forward"), MNN::FeedForward<float, MNN::Activation::Linear>::static_id());
    l.comboType->addItem(tr("convolution"), MNN::Convolution<float, MNN::Activation::Linear>::static_id());

    l.comboAct = new QComboBox(p);
    l.comboAct->addItem(MNN::Activation::Linear::static_name(), MNN::Activation::Linear::static_name());
    l.comboAct->addItem(MNN::Activation::LinearRectified::static_name(), MNN::Activation::LinearRectified::static_name());
    l.comboAct->addItem(MNN::Activation::Tanh::static_name(), MNN::Activation::Tanh::static_name());
    l.comboAct->addItem(MNN::Activation::Logistic::static_name(), MNN::Activation::Logistic::static_name());
    l.comboAct->addItem(MNN::Activation::LogisticSymmetric::static_name(), MNN::Activation::LogisticSymmetric::static_name());
    l.comboAct->addItem(MNN::Activation::Logistic10::static_name(), MNN::Activation::Logistic10::static_name());
    l.comboAct->addItem(MNN::Activation::Sine::static_name(), MNN::Activation::Sine::static_name());
    l.comboAct->addItem(MNN::Activation::Cosine::static_name(), MNN::Activation::Cosine::static_name());
    l.comboAct->addItem(MNN::Activation::Smooth::static_name(), MNN::Activation::Smooth::static_name());
    l.comboAct->addItem(MNN::Activation::Smooth2::static_name(), MNN::Activation::Smooth2::static_name());

    return l;
}
