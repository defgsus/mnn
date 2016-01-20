/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/25/2015</p>
*/

#include <QApplication>

#include "gui/labwindow.h"
#include "gui/analyzewindow.h"
#include "gui/imagelearnwindow.h"

int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    auto mainwin = new ImageLearnWindow();
    mainwin->show();

    return app.exec();
}

