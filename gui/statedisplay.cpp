/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/25/2015</p>
*/

#include <QLayout>
#include <QScrollArea>
#include <QLabel>
#include <QPixmap>
#include <QImage>
#include <QSpinBox>
#include <QDoubleSpinBox>

#include "statedisplay.h"

struct StateDisplay::Private
{
    Private(StateDisplay * p)
        : p                 (p)
        , numInstances      (1)
        , instancesPerRow   (0)
    {
    }

    void createWidgets();
    void setSize(const QSize& s, size_t numInstances);
    void setStates(const float*);
    void setStates(const double*);
    void copyStates();
    void updateStates();
    void updateImage();

    StateDisplay * p;

    QSize size;
    size_t numInstances, instancesPerRow;

    const float * ptr_float;
    const double * ptr_double;
    std::vector<float> states;

    QScrollArea * scrollArea;
    QLabel * labelImg;
    QSpinBox * spinZoom;
    QDoubleSpinBox * spinAmp;
};

StateDisplay::StateDisplay(QWidget *parent)
    : QWidget   (parent)
    , p_        (new Private(this))
{
    p_->createWidgets();

    setStateSize(1, 1, 1);
}

StateDisplay::~StateDisplay()
{
    delete p_;
}

QSize StateDisplay::stateSize() const { return p_->size; }
void StateDisplay::setStateSize(size_t w, size_t h, size_t instances) { p_->setSize(QSize(w, h), instances); }
void StateDisplay::setInstancesPerRow(size_t w) { p_->instancesPerRow = w; }
void StateDisplay::setStateSize(const QSize& s, size_t instances) { p_->setSize(s, instances); }
void StateDisplay::setStates(const float *states) { p_->setStates(states); }
void StateDisplay::setStates(const double *states) { p_->setStates(states); }
void StateDisplay::updateStates() { p_->updateStates(); }

void StateDisplay::Private::createWidgets()
{
    auto lv = new QVBoxLayout(p);
    lv->setMargin(0);

        scrollArea = new QScrollArea(p);
        lv->addWidget(scrollArea);

        labelImg = new QLabel(p);
        scrollArea->setWidget(labelImg);

        auto lh = new QHBoxLayout();
        lv->addLayout(lh);

            spinZoom = new QSpinBox(p);
            spinZoom->setRange(1, 10);
            spinZoom->setValue(1);
            connect(spinZoom, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged),
                    [=]() { updateImage(); });
            lh->addWidget(spinZoom);

            spinAmp = new QDoubleSpinBox(p);
            spinAmp->setRange(0., 1000000000.);
            spinAmp->setValue(1.);
            spinAmp->setSingleStep(0.1);
            connect(spinAmp, static_cast<void(QDoubleSpinBox::*)(double)>
                                (&QDoubleSpinBox::valueChanged), [=]() { updateImage(); });
            lh->addWidget(spinAmp);
}

void StateDisplay::setZoom(int level)
{
    p_->spinZoom->setValue(level);
}

void StateDisplay::Private::setSize(const QSize& s, size_t numI)
{
    if (s == size && numInstances == numI)
        return;
    size = s;
    numInstances = numI;
    ptr_float = 0;
    ptr_double = 0;
    copyStates();
    updateImage();
}

void StateDisplay::Private::setStates(const float *states)
{
    ptr_float = states;
    ptr_double = 0;
    copyStates();
    updateImage();
}

void StateDisplay::Private::setStates(const double *states)
{
    ptr_float = 0;
    ptr_double = states;
    copyStates();
    updateImage();
}

void StateDisplay::Private::updateStates()
{
    if (ptr_float)
        setStates(ptr_float);
    else if (ptr_double)
        setStates(ptr_double);
}

void StateDisplay::Private::copyStates()
{
    states.resize(size.width() * size.height() * numInstances);

    if (ptr_float)
    {
        auto ptr = ptr_float;
        for (auto & s : states)
            s = *ptr++;
    }
    else if (ptr_double)
    {
        auto ptr = ptr_double;
        for (auto & s : states)
            s = *ptr++;
    }
    else
    {
        for (auto & s : states)
            s = 0.;
    }
}

void StateDisplay::Private::updateImage()
{
    if (size.isNull())
    {
        labelImg->clear();
        return;
    }

    // determine image size
    QSize si = size;
    if (numInstances > 1)
    {
        size_t in = numInstances;
        if (instancesPerRow > 0)
        {
            in = std::min(in, instancesPerRow);
            si.setHeight(si.height() * std::max(size_t(1),
                            (numInstances+instancesPerRow-1) / instancesPerRow) );
        }
        si.setWidth(si.width() * in);
    }

    // prepare image
    QImage img(si, QImage::Format_ARGB32);
    img.fill(Qt::black);

    // plot states
    const float amp = spinAmp->value();
    float * s = &states[0];
    int ix = 0, iy = 0;
    for (size_t i = 0; i < numInstances; ++i)
    {
        if (0)
        {
            for (int y = 0; y < size.height(); ++y)
            for (int x = 0; x < size.width(); ++x, ++s)
            {
                int si = std::min(255, std::abs(int(*s * 255 * amp)));
                int si0 = si * 0.9;
                img.setPixel(ix * size.width() + x,
                             iy * size.height() + y,
                             *s > 0 ? qRgb(si, si, si0) : qRgb(si0, si0, si));
            }
        }
        else
        {
            for (int y = 0; y < size.height(); ++y)
            for (int x = 0; x < size.width(); ++x, ++s)
            {
                int si = std::min(255, std::abs(int((*s * amp + .5f) * 255)));
                img.setPixel(ix * size.width() + x,
                             iy * size.height() + y,
                             qRgb(si, si, si));
            }
        }
        ++ix;
        if (instancesPerRow > 0 && ix >= (int)instancesPerRow)
        {
            ix = 0;
            ++iy;
        }
    }

    labelImg->setPixmap(QPixmap::fromImage(img.scaled(img.size() * spinZoom->value())));
    labelImg->adjustSize();
}

