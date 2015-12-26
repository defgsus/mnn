#-------------------------------------------------
#
# Project created by QtCreator 2014-05-09T22:58:26
#
#-------------------------------------------------

QT       += core gui widgets

TARGET = devnn_gui
CONFIG   += c++11

TEMPLATE = app


SOURCES += \
    main_gui.cpp \
    trainposition.cpp \
    mnistset.cpp \
    trainmnist.cpp \
    mnn/stack_serial_impl.inl \
    mnn/perceptron_impl.inl \
    mnn/perceptronbias_impl.inl \
    mnn/rbm_impl.inl \
    gui/labwindow.cpp \
    gui/statedisplay.cpp \
    gui/trainthread.cpp

HEADERS += \
    trainposition.h \
    trainmnist.h \
    mnn/activation.h  \
    mnn/function.h  \
    mnn/layer.h  \
    mnn/mnn.h  \
    mnn/perceptron.h  \
    mnn/stack_serial.h  \
    mnistset.h \
    mnn/perceptronbias.h \
    mnn/rbm.h \
    mnn/exception.h \
    gui/labwindow.h \
    gui/statedisplay.h \
    gui/rbm_stack.h \
    gui/trainthread.h

OTHER_FILES += \
    .gitignore \
    README.md
