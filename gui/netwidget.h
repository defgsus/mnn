/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/21/2016</p>
*/

#ifndef NETWIDGET_H
#define NETWIDGET_H

#include <QWidget>

/** Widget to setup a network */
class NetWidget : public QWidget
{
    Q_OBJECT
public:
    explicit NetWidget(QWidget *parent = 0);
    ~NetWidget();

signals:

public slots:

private:

    struct Private;
    Private * p_;
};

#endif // NETWIDGET_H
