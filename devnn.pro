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


SOURCES += main.cpp \
    trainposition.cpp \
    mnistset.cpp \
    trainmnist.cpp

HEADERS += \
    mnn/activation.h  \
    mnn/function.h  \
    mnn/layer.h  \
    mnn/mnn.h  \
    mnn/perceptron.h  \
    mnn/perceptron_impl.inl  \
    mnn/stack_serial.h  \
    mnn/stack_serial_impl.inl  \
    trainposition.h \
    mnistset.h \
    trainmnist.h
