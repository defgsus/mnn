/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/25/2015</p>
*/

#ifndef LABWINDOW_H
#define LABWINDOW_H

#include <QMainWindow>

class LabWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit LabWindow(QWidget *parent = 0);
    ~LabWindow();

signals:

public slots:

private:
    struct Private;
    Private * p_;
};

#endif // LABWINDOW_H
