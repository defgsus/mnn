#-------------------------------------------------
#
# Project created by QtCreator 2014-05-09T22:58:26
#
#-------------------------------------------------

QT       += core gui widgets

TARGET = devnn_gui
CONFIG   += c++11

TEMPLATE = app

include(mnn/mnn.pri)

SOURCES += \
    main_gui.cpp \
    trainposition.cpp \
    mnistset.cpp \
    trainmnist.cpp \
    cifarset.cpp \
    gui/labwindow.cpp \
    gui/statedisplay.cpp \
    gui/trainthread.cpp \
    gui/analyzewindow.cpp \
    image.cpp \
    gui/imagelearnwindow.cpp \
    imagelearner.cpp \
    imagelearnthread.cpp

HEADERS += \
    trainposition.h \
    trainmnist.h \
    mnistset.h \
    printstate.h \
    generate_input.h \
    cifarset.h \
    gui/labwindow.h \
    gui/statedisplay.h \
    gui/rbm_stack.h \
    gui/trainthread.h \
    gui/analyzewindow.h \
    image.h \
    gui/imagelearnwindow.h \
    imagelearner.h \
    imagelearnthread.h

OTHER_FILES += \
    .gitignore \
    README.md
