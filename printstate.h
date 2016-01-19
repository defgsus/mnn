/** @file

    @brief

    <p>(c) 2015, stefan.berke@modular-audio-graphics.com</p>
    <p>All rights reserved</p>

    <p>created 12/26/2015</p>
*/

#ifndef PRINTSTATE_H
#define PRINTSTATE_H

#include <iostream>

template <typename F>
void printState(const F* state, size_t width, size_t height = 1,
                std::ostream& out = std::cout)
{
    for (size_t j=0; j<height; ++j)
    {
        for (size_t i=0; i<width; ++i, ++state)
        {
            out << *state << " ";
        }
        out << std::endl;
    }
}

template <typename F>
void printStateAscii(const F* state, size_t width, size_t height,
                     F amplitude = 1, std::ostream& out = std::cout)
{
    for (size_t j=0; j<height; ++j)
    {
        for (size_t i=0; i<width; ++i, ++state)
        {
            F s = *state * amplitude;
            out << ( s > .7 ? '#' : s > .35 ? '*' : s > .15 ? ':' : '.' );
        }
        out << std::endl;
    }
}

template <typename F>
void printStateAscii(const F* state, size_t width, size_t height, size_t instances,
                     F amplitude = 1, std::ostream& out = std::cout, size_t maxCols = 230)
{
    for (size_t j=0; j<height; ++j)
    {
        size_t col = 0;
        for (size_t ins=0; ins<instances; ++ins)
        {
            if (col >= maxCols)
                break;
            for (size_t i=0; i<width; ++i, ++col)
            {
                if (col >= maxCols)
                    break;
                F s = state[(j * width + i) + ins * width * height] * amplitude;
                out << ( s > .7 ? '#' : s > .35 ? '*' : s > .15 ? ':' : '.' );
            }
            out << " ";
            ++col;
        }
        out << std::endl;
    }
}

#endif // PRINTSTATE_H

