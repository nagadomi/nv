/*
 * This file is part of libnv.
 *
 * Copyright (C) 2008-2012 nagadomi@nurs.or.jp
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
 * This is based on public domain SHA1 implementation.
 * To get the original version,
 * contact <http://www.aarongifford.com/computers/sha.html>
 * The original copyright notice follows.
 *
 * Originally taken from the public domain SHA1 implementation
 * written by by Steve Reid <steve@edmweb.com>
 * 
 * Modified by Aaron D. Gifford <agifford@infowest.com>
 *
 * NO COPYRIGHT - THIS IS 100% IN THE PUBLIC DOMAIN
 *
 * The original unmodified version is available at:
 *    ftp://ftp.funet.fi/pub/crypt/hash/sha/sha1.c
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR(S) OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef NV_SHA1_H
#define NV_SHA1_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __BIG_ENDIAN__
#  undef LITTLE_ENDIAN
#else
#  ifndef LITTLE_ENDIAN
#    define LITTLE_ENDIAN
#  endif
#endif	
#include "nv_core.h"

/* Make sure you define these types for your architecture: */
typedef uint32_t sha1_quadbyte;	/* 4 byte type */
typedef uint8_t sha1_byte;	/* single byte type */

/*
 * Be sure to get the above definitions right.  For instance, on my
 * x86 based FreeBSD box, I define LITTLE_ENDIAN and use the type
 * "unsigned long" for the quadbyte.  On FreeBSD on the Alpha, however,
 * while I still use LITTLE_ENDIAN, I must define the quadbyte type
 * as "unsigned int" instead.
 */

#define SHA1_BLOCK_LENGTH	64
#define SHA1_DIGEST_LENGTH	20

/* The SHA1 structure: */
typedef struct _SHA_CTX {
	sha1_quadbyte	state[5];
	sha1_quadbyte	count[2];
	sha1_byte	buffer[SHA1_BLOCK_LENGTH];
} SHA_CTX;

void sha1_init(SHA_CTX *context);
void sha1_update(SHA_CTX *context, const sha1_byte *data, unsigned int len);
void sha1_final(sha1_byte *digest, SHA_CTX* context);

#ifdef	__cplusplus
}
#endif

#endif

