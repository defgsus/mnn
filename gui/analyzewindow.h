/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/15/2016</p>
*/

#ifndef ANALYZEWINDOW_H
#define ANALYZEWINDOW_H

#include <QMainWindow>

class AnalyzeWindow : public QMainWindow
{
    Q_OBJECT
public:
    AnalyzeWindow(QWidget* parent = 0, Qt::WindowFlags flags = 0);
    ~AnalyzeWindow();

public slots:
    void openLayer();

private:
    struct Private;
    Private* p_;
};

#endif // ANALYZEWINDOW_H
