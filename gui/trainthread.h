/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/25/2015</p>
*/

#ifndef TRAINTHREAD_H
#define TRAINTHREAD_H

#include <QObject>
#include <QThread>

#include "rbm_stack.h"

typedef float Float;

class QReadWriteLock;

class TrainThread : public QThread
{
    Q_OBJECT
public:
    typedef MNN::Rbm<Float, MNN::Activation::Logistic> NetType;
    typedef RbmStack<Float, NetType> StackType;

    TrainThread(QObject * parent);
    ~TrainThread();

    const StackType& rbmStack() const { return rbms_; }
    // locked access
    NetType getNetCopy(size_t index) const;

public slots:

    void setSize(const std::vector<size_t>& sizes) { sizeRequest_ = sizes; }

    void stop() { doRun_ = false; }
    void setPause(bool enable) { doPause_ = enable; }

signals:

    void rbmChanged();

protected:

    void run() override;

    StackType rbms_;

    volatile bool doRun_, doPause_;
    volatile size_t trainLayerIndex_;
    std::vector<size_t> sizeRequest_;
    QReadWriteLock * mutex_;
};

#endif // TRAINTHREAD_H
