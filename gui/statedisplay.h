/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/25/2015</p>
*/

#ifndef STATEDISPLAY_H
#define STATEDISPLAY_H

#include <QWidget>

/** A generic display for float/double states.
    Can be one or two dimensional. */
class StateDisplay : public QWidget
{
    Q_OBJECT
public:
    explicit StateDisplay(QWidget *parent = 0);
    ~StateDisplay();

    QSize stateSize() const;

public slots:

    void setZoom(int level);

    /** Sets the size/dimension of the display.
        Any pointers given to setStates are dropped */
    void setStateSize(size_t width, size_t height = 1, size_t instances = 1);
    void setStateSize(const QSize& s, size_t instances = 1);
    /** Sets the number of multiple instances that are maximally
        placed in one row. Use 0 to don't care. */
    void setInstancesPerRow(size_t w);

    /** Copies the values in @p states assuming the currently
        given size. The pointer given is not longer referenced
        unless updateStates() is called. */
    void setStates(const float * states);
    void setStates(const double * states);

    /** Rereads the states previously from the pointer
        previously given to setStates() */
    void updateStates();

private:
    struct Private;
    Private * p_;
};

#endif // STATEDISPLAY_H
