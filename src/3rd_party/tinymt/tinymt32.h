#ifndef TINYMT32_H
#define TINYMT32_H
/*
 * This file is part of libnv.
 *
 * Copyright (C) 2014 nagadomi@nurs.or.jp
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License,
 * or any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/*
 * improve portability
 */
/*
  This is based on tinymt32.  To get the original version,
  contact <http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/>.
  
  The original copyright notice follows.
*/
/**
 * @file tinymt32.h
 *
 * @brief Tiny Mersenne Twister only 127 bit internal state
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (University of Tokyo)
 *
 * Copyright (C) 2011 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see
 * LICENSE.txt
 */

#include <stdint.h>
#include <inttypes.h>

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * tinymt32 internal state vector and parameters
 */
typedef struct TINYMT32_T {
    uint32_t status[4];
    uint32_t mat1;
    uint32_t mat2;
    uint32_t tmat;
} tinymt32_t;

void tinymt32_init_param(tinymt32_t * random,
						 uint32_t mat1, uint32_t mat2, uint32_t tmat);
void tinymt32_init(tinymt32_t * random, uint32_t seed);
void tinymt32_init_by_array(tinymt32_t * random, uint32_t init_key[],
			    int key_length);

uint32_t tinymt32_generate_uint32(tinymt32_t * random);
float tinymt32_generate_float(tinymt32_t * random);
float tinymt32_generate_float12(tinymt32_t * random);
float tinymt32_generate_float01(tinymt32_t * random);
float tinymt32_generate_floatOC(tinymt32_t * random);
float tinymt32_generate_floatOO(tinymt32_t * random);
double tinymt32_generate_32double(tinymt32_t * random);

#if defined(__cplusplus)
}
#endif

#endif
