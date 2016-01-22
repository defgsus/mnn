/** @file

    @brief

    <p>(c) 2016, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 1/20/2016</p>
*/

#ifndef IMAGELEARNWINDOW_H
#define IMAGELEARNWINDOW_H

#include <QMainWindow>

class ImageLearnWindow : public QMainWindow
{
    Q_OBJECT
public:

    typedef float Float;

    explicit ImageLearnWindow(QWidget *parent = 0);
    ~ImageLearnWindow();

signals:

public slots:

    void loadImage();
    void loadImage(const QString& fn);
    void loadCorruptImage();
    void loadCorruptImage(const QString& fn);

    void newNetwork();
    void loadNetwork();
    void loadNetwork(const QString& fn);
    void saveNetwork();
    void saveNetwork(const QString& fn);

    void renderImageReconstruction();
    void renderOrgImageReconstruction();

    void startThread();
    void stopThread();

private slots:

    void onProgress();
    void onReconstruction();

private:
    struct Private;
    Private* p_;
};

#endif // IMAGELEARNWINDOW_H
