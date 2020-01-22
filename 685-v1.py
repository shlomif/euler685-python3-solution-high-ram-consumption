#!/usr/bin/env python

# The Expat License
#
# Copyright (c) 2019, Shlomi Fish
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from functools import reduce
import functools
import pprint
import sys

import click

# from memory_profiler import profile
# import six

try:
    import numpy as np
except BaseException:
    class np:
        pass

    def np_zeros(x, dtype):
        return [[0, 0], [0, 0]]
    np.int64 = 0
    np.zeros = np_zeros

from six import print_

import sparse_list


DEBUG = 1
pprint.pprint("")


def digit_sum(n):
    return sum([int(d) for d in str(n)])


def brute_find_lowest_num_with_digit_sum(n):
    ret = 1
    while True:
        if digit_sum(ret) == n:
            return ret
        ret += 1


def brute_calc_f(n, m):
    ret = 1
    while True:
        if digit_sum(ret) == n:
            m -= 1
            if not m:
                return ret
        ret += 1


def not_so_brute_calc_f__calc_iter(n):
    def myiter(L, prefix, digsum):
        if len(prefix) == L:
            if digsum == n:
                yield prefix
            return
        if digsum > n:
            return
        top = digsum + (L-len(prefix))*9
        if top < n:
            return
        for d in range((len(prefix) == 0), 10):
            for x in myiter(L, prefix+[d], digsum+d):
                yield x

    def l_iter():
        L = 0
        while True:
            it = myiter(L, [], 0)
            for x in it:
                yield x
            L += 1
    it = l_iter()
    return it


def not_so_brute_calc_f(n, m):
    it = not_so_brute_calc_f__calc_iter(n)
    mm = m
    for x in it:
        mm -= 1
        if mm == 0:
            return int("".join([str(y) for y in x]))


def brute_count(n, bottom, top):
    r = 0
    for x in range(int(bottom), int(top)):
        if digit_sum(x) == n:
            r += 1
    return r


def mod_matmul(MOD, n, m1, m2):
    ret = np.zeros((n, n), dtype=np.int64)
    for y in range(n):
        row = m1[y]
        for x in range(n):
            ret[y][x] = sum(int(row[i])*int(m2[i][x]) for i in
                            range(n)) % MOD
    return ret

# Planning:
# Row 0 - the constant 1
# Row 1 - "1, 19, 199, 1999" ;
# "2, 29, 299, 2999" / etc.
# Row 2 - the sum


def compare(got, expected, blurb):
    if got != expected:
        print_('blurb =', blurb, 'got =', got, 'expected =', expected)
    assert got == expected


@functools.lru_cache(maxsize=10000)
def nCr(n, k):
    if k > (n >> 1):
        return nCr(n, n-k)
    r = 1
    for x in range(n-k + 1, n+1):
        r *= x
    d = 1
    for x in range(2, k+1):
        d *= x
    # assert r % d == 0
    r //= d
    # compare(r, FACTS[n] // FACTS[k] // FACTS[n-k],
    #         "ncr(k={};n={})".format(k, n))
    return r


ctr = 0
count9 = 0
non9 = 0


def _get_initial_by_digit():
    return [[], [], [], [], [], [], [], [], [], [], ]


def _get_initial_0s():
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]


def listsum(lst):
    ret = 0
    for x in lst:
        assert type(x) == int
        ret += x
    return ret


class MatrixMod:
    DIGITS_CYCLE = 9

    def __init__(self, MOD):
        self.MOD = MOD
        self.m1 = self.gen_matrix()
        self.range9 = range(self.DIGITS_CYCLE)
        self._count_total_cache = {}

    def gen_matrix(self):
        L = 2
        ret = np.zeros((L, L), dtype=np.int64)

        def assign(from_, to, val):
            ret[to][from_] = val

        digit = 0
        mysum = 1
        assign(digit, mysum, 9)
        assign(digit, digit, 1)
        assign(mysum, mysum, 10)

        return ret

    def mass_check(self):
        range_sum = 0
        for x in range(1, 1000):
            term = brute_find_lowest_num_with_digit_sum(x)
            print_("s({}) = {}".format(x, term),
                   flush=True)
            range_sum += term
            compare(
                self.calc_s(x),
                range_sum % self.MOD, "calc_s[{}]".format(x)
            )

    def calc_s(self, n):
        return sum(
            [self.calc_s_for_uppermost_digit(uppermost_digit, n)
             for uppermost_digit in range(self.DIGITS_CYCLE)]
        ) % self.MOD

    # We calculate the sums for:
    #   [uppermost_digit = 0]: 9 + 99 + 999 + 9999 ...
    #   [uppermost_digit = 1]: 1 + 19 + 199 + 1999 + 19999 ...
    #   2: 2 + 29 + 299 + 2999 + 29999 ...
    # Etc. separately.
    def calc_s_for_uppermost_digit(self, uppermost_digit, n):
        n_mod = n
        while n_mod % self.DIGITS_CYCLE != uppermost_digit:
            n_mod -= 1
        e = n_mod // self.DIGITS_CYCLE
        if e < 0:
            return 0
        if e == 0:
            return uppermost_digit
        mat = self.mat_exp_mod(e)
        return (mat[2, 0] + (int(mat[2, 1]) + int(mat[2, 2]))
                * uppermost_digit)

    def mat_exp_mod(self, e):
        m1 = self.m1
        if e == 1:
            return m1
        sub = self.mat_exp_mod(e >> 1)
        ret = mod_matmul(self.MOD, 2, sub, sub)
        return mod_matmul(self.MOD, 2, ret, m1) if e & 1 else ret

    def _gen_01_L_initial_sequences(self, L, digit_sum):
        ret = 0
        factors = [1]

        def rec(ll, rem):
            if ll > L:
                return
            padded = rem + 9 * (L - ll)
            if padded < digit_sum:
                return
            if rem > digit_sum:
                return
            is_valid = padded == digit_sum
            if is_valid:
                while ll >= len(factors):
                    factors.append(nCr(L, len(factors)))
                nonlocal ret
                ret += factors[ll]
            for i in self.range9:
                rec(ll+1, rem + i)

        rec(0, 0)
        return ret

    def _gen_L_initial_sequences__old(self, L, digit_sum):
        sequences1 = [[] for _ in range(L+1)]
        sequences0 = [[] for _ in range(L+1)]

        def rec(so_far, rem):
            ll = len(so_far)
            if ll > L:
                return
            padded = rem + 9 * (L - ll)
            if padded < digit_sum:
                return
            if rem > digit_sum:
                return
            is_valid = padded == digit_sum
            if is_valid:
                if ll and so_far[0] == 0:
                    sequences0[ll].append(so_far)
                else:
                    sequences1[ll].append(so_far)
            for i in self.range9:
                rec(so_far + [i], rem + i)

        rec([], 0)
        seqs = (sequences0, sequences1)
        return seqs

    def _gen_L_initial_sequences(self, L, digit_sum):
        sequences1 = [_get_initial_0s() for _ in range(L+1)]
        sequences0 = [_get_initial_0s() for _ in range(L+1)]

        def rec(ll, first, rem):
            if ll > L:
                return
            padded = rem + 9 * (L - ll)
            if padded < digit_sum:
                return
            if rem > digit_sum:
                return
            is_valid = padded == digit_sum
            if is_valid:
                if ll:
                    if first == 0:
                        sequences0[ll][first] += 1
                    else:
                        sequences1[ll][first] += 1
                else:
                    sequences1[ll][0] += 1
            for i in self.range9:
                rec(ll+1, (i if not ll else first), rem + i)

        rec(0, None, 0)
        seqs = (sequences0, sequences1)
        return seqs

    def calc_f(self, n, m):
        remainder = n % self.DIGITS_CYCLE
        count9s = n // self.DIGITS_CYCLE
        L = count9s+1 if remainder else count9s
        remaining_m = m
        if m == 1:
            return int(str(remainder) + "9"*count9s)
        while remaining_m:
            if remainder == 0:
                remainder += self.DIGITS_CYCLE
                count9s -= 1
            new_seqs = self._gen_L_initial_sequences(L, n)
            # print_("seqs0={} seqs1={}".format(sequences0, sequences1))
            total_diff = 0
            for leading_digit in range(2):
                digit_l = L - 1 + leading_digit
                news = new_seqs[leading_digit]
                c = 0
                for seq_len in range(digit_l + 1):
                    count = sum(news[seq_len])
                    if count:
                        total_diff += count * nCr(digit_l, seq_len)
                        if c == 1:
                            c = 2
                    else:
                        if not c:
                            c = 1
                        elif c == 2:
                            # compare(seq_len, 0, 'seq_len')
                            break
            newfound = remaining_m - total_diff
            if newfound > 0:
                remaining_m = newfound
                L += 1
            else:
                ret = self._calc_f_finetune(L, new_seqs, n, remaining_m)
                self._nullify_seq()
                return ret

    def _nullify_seq(self):
        self.newzs = None
        self._new_seq_counts = None

    def _set_seq(self, new_val):
        self.newzs = new_val
        # print_(self.zs)
        self._new_seq_counts = \
            [sum(self.newzs[i]) for i in self.nrange]

    def _calc_f_finetune(self, L, new_seqs, n, remaining_m):
        global count9
        global non9

        prefix = sparse_list.SparseList(0, 9)
        new_seqs_incremental = []
        ncrl = L-1
        self.nrange = range(ncrl+1)
        cant_start_with_zero = 1
        zrange = range(cant_start_with_zero, 9)
        digsum = 0

        def merge(dest_seq_by_len):
            sequences = [_get_initial_0s() for _ in range(L-len(prefix)+1)]
            for s in dest_seq_by_len:
                for ll, digits in enumerate(s):
                    for dd, v in enumerate(digits):
                        sequences[ll][dd] += v
            return sequences

        for news in new_seqs:
            new_dest_seq_by_len = []
            for ll, newx in enumerate(news):
                if ll == 0:
                    continue
                new_dest_len_seq__by_digit = _get_initial_0s()
                for dd, y in enumerate(newx):
                    new_dest_len_seq__by_digit[dd] += y
                new_dest_seq_by_len.append(new_dest_len_seq__by_digit)
            new_seqs_incremental.append(new_dest_seq_by_len)
        self._set_seq(new_seqs_incremental[1])

        def descend_several_9s(count_new_9s):
            nonlocal digsum
            nonlocal ncrl
            prefix.size += count_new_9s
            digsum += count_new_9s * 9
            ncrl -= count_new_9s
            self.nrange = range(ncrl+1)

        def descend(d):
            nonlocal cant_start_with_zero
            nonlocal digsum
            nonlocal ncrl
            if cant_start_with_zero:
                cant_start_with_zero = 0
                nonlocal zrange
                zrange = self.range9
                new_dest_seq_by_len = merge(new_seqs_incremental)
                self._set_seq(new_dest_seq_by_len)
            prefix.append(d)
            digsum += d
            ncrl -= 1
            self.nrange = range(ncrl+1)
            # pprint.pprint([prefix.elements, prefix.size])
            if d < 9:
                new_dest_seq_by_len = self._gen_L_initial_sequences(
                    L-len(prefix), n-digsum)
                new_sequences = merge(new_dest_seq_by_len)
                new_sequences.pop(0)
                self._set_seq(new_sequences)

        while True:
            # pprint.pprint(locals())
            def count_perms(ll, lenitem):

                def _c():
                    if lenitem == 0:
                        return 1
                    if lenitem == 1:
                        return ncrl
                    return nCr(ncrl, lenitem)
                if ll:
                    return _c() * ll
                return 0

            remainlen = L - len(prefix)
            assert remainlen >= 0
            maxdigsum = remainlen * 9 + digsum
            digitcond = maxdigsum == n+1
            if maxdigsum == n or digitcond:
                if digitcond:
                    prefix.size += (remaining_m-1)
                    prefix.append(8)
                    prefix.size += (remainlen - remaining_m)
                else:
                    prefix.size += remainlen
                assert prefix.size == L
                keys = sorted(prefix.elements.keys())
                reached = 0
                ret = 0

                def _advance_reached(place):
                    nonlocal reached
                    nonlocal ret
                    if reached >= place:
                        return
                    mat = self.mat_exp_mod(place - reached)
                    ret = (mat[1][0] + ret * mat[1][1]) % self.MOD
                    reached = place

                for place in keys:
                    _advance_reached(place)
                    ret = (ret * 10 + prefix[place]) % self.MOD
                    reached += 1
                _advance_reached(prefix.size)
                # ret = ''.join(prefix)
                return int(ret)

            mydiff8 = sum(
               count_perms(
                   self._new_seq_counts[i],
                   i,
               ) for i in self.nrange
            )
            if mydiff8 < remaining_m:
                use_one = True
                if len(prefix) and remainlen:

                    def _count_total(remain_digsum, suffix_len):
                        key = (remain_digsum, suffix_len)
                        if key in self._count_total_cache:
                            print_('hit', key)
                            return self._count_total_cache[key]
                        ret = self._gen_01_L_initial_sequences(
                             suffix_len,
                             remain_digsum,
                        )
                        self._count_total_cache[key] = ret
                        return ret
                    expected_before = _count_total(
                        n - digsum, remainlen
                    )
                    if 0:
                        got = sum([self._new_seq_counts[i]*nCr(remainlen, i+1)
                                   for i in range(min([
                                       1+remainlen,
                                       len(self._new_seq_counts)]))])
                        compare(expected=got,
                                got=expected_before,
                                blurb="_new_seq_counts")
                    top = remainlen
                    bottom = 1
                    while use_one and (top-bottom > 1):
                        remainlen_to_check = (top+bottom) >> 1
                        count_new_9s = remainlen - remainlen_to_check
                        new_digsum = digsum + count_new_9s * 9
                        new_remain_digsum = n - new_digsum
                        if new_remain_digsum <= 0:
                            top = remainlen_to_check-1
                        else:
                            expected_after = _count_total(
                                new_remain_digsum, remainlen_to_check)
                            expected = expected_before - expected_after
                            if expected < remaining_m:
                                remaining_m -= expected
                                descend_several_9s(count_new_9s)
                                print_('count_new_9s', count_new_9s)
                                use_one = False
                            else:
                                bottom = remainlen_to_check + 1
                                # print_('new bottom =', bottom, expected)
                if use_one:
                    remaining_m -= mydiff8
                    descend(9)
                count9 += 1
                # print_('count9 =', count9, 'non9 =', non9)
            else:
                non9 += 1
                for d in zrange:
                    mydiff = sum(
                        count_perms(self.newzs[i][d], i)
                        for i in self.nrange
                    )
                    if mydiff < remaining_m:
                        remaining_m -= mydiff
                    else:
                        descend(d)
                        break


@click.command()
@click.option(
    '--top', default=10000,
    help='calculate the sum of f(x**3,x**4) to what limit')
@click.option(
    '--bottom', default=1,
    help='calculate the sum of f(x**3,x**4) starting from what limit')
def main(top, bottom):
    # top = int(sys.argv.pop(1)) if len(sys.argv) >= 2 else 10000
    mat = MatrixMod(10 ** 9 + 7)
    s = 0
    if 0:
        for k in [20]:
            n = k ** 3
            m = n * k
            print_('before', k, n, m)
            sys.stdout.flush()
            ret = mat.calc_f(n, m)
            print_('after', n, m)
            return
    if 1:
        for nn in range(2, 28):
            nn_iter = not_so_brute_calc_f__calc_iter(nn)
            for k in range(1, 11):
                n = nn
                m = k
                ret = mat.calc_f(n, m)
                good_ret = int("".join([str(y) for y in next(nn_iter)]))
                # good_ret = ret
                compare(ret, good_ret, "calc_f({},{})".format(n, m))

    k_checkpoint = 10
    if bottom == 1:
        for k in range(1, k_checkpoint + 1):
            n = k ** 3
            m = n * k
            print_('before', n, m)
            sys.stdout.flush()
            ret = mat.calc_f(n, m)
            print_('after', n, m, ret)
            sys.stdout.flush()
            good_ret = not_so_brute_calc_f(n, m)
            good_ret_mod = good_ret % mat.MOD
            # good_ret = ret
            compare(
                ret, good_ret_mod, "calc_f(k={};n={};m={})".format(k, n, m))
            sys.stdout.flush()
            s += ret
            sys.stdout.flush()
            if k == 3:
                compare(s, 7128, "S(3)")
    else:
        k_checkpoint = bottom - 1

    s %= mat.MOD
    compare(s, 32287064, "s[10]")
    for k in range(k_checkpoint+1, top+1):
        n = k ** 3
        m = n * k
        print_('before', k, n, m)
        sys.stdout.flush()
        ret = mat.calc_f(n, m)
        print_('after', n, m)
        sys.stdout.flush()
        # good_ret = not_so_brute_calc_f(n, m)
        # good_ret = ret
        # compare(ret, good_ret, "calc_f(k={};n={};m={})".format(k, n, m))
        # sys.stdout.flush()
        s += ret
        s %= mat.MOD
        print_('after s', k, s)
        sys.stdout.flush()

    print_("Final result = {}".format(s))
    return
    fibs = [1, 2]
    fibs_sum = 0
    for fibs_idx in range(2, 90 + 1):
        fibs_sum += mat.calc_s(fibs[0])
        fibs = [fibs[1], sum(fibs)]
    print_("result = {}".format(fibs_sum % mat.MOD))
    return


if __name__ == "__main__":
    main()
