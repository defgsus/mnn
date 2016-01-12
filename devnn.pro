#-------------------------------------------------
#
# Project created by QtCreator 2014-05-09T22:58:26
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = devnn
CONFIG   += console c++11
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += \
    main.cpp \
    trainposition.cpp \
    mnistset.cpp \
    trainmnist.cpp \
    mnn/stack_serial_impl.inl \
    mnn/perceptron_impl.inl \
    mnn/perceptronbias_impl.inl \
    mnn/rbm_impl.inl \
    mnn/convolution_impl.inl

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
    printstate.h \
    mnn/convolution.h

OTHER_FILES += \
    .gitignore \
    README.md
