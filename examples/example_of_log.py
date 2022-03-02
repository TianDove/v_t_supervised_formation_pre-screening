#！user/bin/env python
# -*- coding: utf-8 -*-
# author: Wang, Xiang

def foo1(a, b, c=None, d=None, f=None, **kw):
    x = a + b

    print(x, c, d, f)


def foo2(a, *, b):
    x = a + b
    print(x)


if __name__ == '__main__':
    foo1(1, 2, {'c': 1, 'b': 2, 'f': 3}, t = 1231,p = 12312,k = 1231)