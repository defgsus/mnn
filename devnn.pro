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

include(mnn/mnn.pri)

SOURCES += \
    main.cpp \
    trainposition.cpp \
    mnistset.cpp \
    trainmnist.cpp \
    cifarset.cpp

HEADERS += \
    trainposition.h \
    trainmnist.h \
    mnistset.h \
    printstate.h \
    generate_input.h \
    cifarset.h

OTHER_FILES += \
    .gitignore \
    README.md
