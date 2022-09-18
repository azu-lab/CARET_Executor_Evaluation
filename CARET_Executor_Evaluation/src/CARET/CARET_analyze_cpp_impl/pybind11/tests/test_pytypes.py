# -*- coding: utf-8 -*-
from __future__ import division

import sys

import pytest

import env  # noqa: F401
from pybind11_tests import debug_enabled
from pybind11_tests import pytypes as m


def test_int(doc):
    assert doc(m.get_int) == "get_int() -> int"


def test_iterator(doc):
    assert doc(m.get_iterator) == "get_iterator() -> Iterator"


def test_iterable(doc):
    assert doc(m.get_iterable) == "get_iterable() -> Iterable"


def test_list(capture, doc):
    with capture:
        lst = m.get_list()
        assert lst == ["inserted-0", "overwritten", "inserted-2"]

        lst.append("value2")
        m.print_list(lst)
    assert (
        capture.unordered
        == """
        Entry at position 0: value
        list item 0: inserted-0
        list item 1: overwritten
        list item 2: inserted-2
        list item 3: value2
    """
    )

    assert doc(m.get_list) == "get_list() -> list"
    assert doc(m.print_list) == "print_list(arg0: list) -> None"


def test_none(capture, doc):
    assert doc(m.get_none) == "get_none() -> None"
    assert doc(m.print_none) == "print_none(arg0: None) -> None"


def test_set(capture, doc):
    s = m.get_set()
    assert s == {"key1", "key2", "key3"}

    with capture:
        s.add("key4")
        m.print_set(s)
    assert (
        capture.unordered
        == """
        key: key1
        key: key2
        key: key3
        key: key4
    """
    )

    assert not m.set_contains(set(), 42)
    assert m.set_contains({42}, 42)
    assert m.set_contains({"foo"}, "foo")

    assert doc(m.get_list) == "get_list() -> list"
    assert doc(m.print_list) == "print_list(arg0: list) -> None"


def test_dict(capture, doc):
    d = m.get_dict()
    assert d == {"key": "value"}

    with capture:
        d["key2"] = "value2"
        m.print_dict(d)
    assert (
        capture.unordered
        == """
        key: key, value=value
        key: key2, value=value2
    """
    )

    assert not m.dict_contains({}, 42)
    assert m.dict_contains({42: None}, 42)
    assert m.dict_contains({"foo": None}, "foo")

    assert doc(m.get_dict) == "get_dict() -> dict"
    assert doc(m.print_dict) == "print_dict(arg0: dict) -> None"

    assert m.dict_keyword_constructor() == {"x": 1, "y": 2, "z": 3}


def test_tuple():
    assert m.get_tuple() == (42, None, "spam")


@pytest.mark.skipif("env.PY2")
def test_simple_namespace():
    ns = m.get_simple_namespace()
    assert ns.attr == 42
    assert ns.x == "foo"
    assert ns.right == 2
    assert not hasattr(ns, "wrong")


def test_str(doc):
    assert m.str_from_string().encode().decode() == "baz"
    assert m.str_from_bytes().encode().decode() == "boo"

    assert doc(m.str_from_bytes) == "str_from_bytes() -> str"

    class A(object):
        def __str__(self):
            return "this is a str"

        def __repr__(self):
            return "this is a repr"

    assert m.str_from_object(A()) == "this is a str"
    assert m.repr_from_object(A()) == "this is a repr"
    assert m.str_from_handle(A()) == "this is a str"

    s1, s2 = m.str_format()
    assert s1 == "1 + 2 = 3"
    assert s1 == s2

    malformed_utf8 = b"\x80"
    if hasattr(m, "PYBIND11_STR_LEGACY_PERMISSIVE"):
        assert m.str_from_object(malformed_utf8) is malformed_utf8
    elif env.PY2:
        with pytest.raises(UnicodeDecodeError):
            m.str_from_object(malformed_utf8)
    else:
        assert m.str_from_object(malformed_utf8) == "b'\\x80'"
    if env.PY2:
        with pytest.raises(UnicodeDecodeError):
            m.str_from_handle(malformed_utf8)
    else:
        assert m.str_from_handle(malformed_utf8) == "b'\\x80'"

    assert m.str_from_string_from_str("this is a str") == "this is a str"
    ucs_surrogates_str = u"\udcc3"
    if env.PY2:
        assert u"\udcc3" == m.str_from_string_from_str(ucs_surrogates_str)
    else:
        with pytest.raises(UnicodeEncodeError):
            m.str_from_string_from_str(ucs_surrogates_str)


def test_bytes(doc):
    assert m.bytes_from_string().decode() == "foo"
    assert m.bytes_from_str().decode() == "bar"

    assert doc(m.bytes_from_str) == "bytes_from_str() -> {}".format(
        "str" if env.PY2 else "bytes"
    )


def test_bytearray(doc):
    assert m.bytearray_from_string().decode() == "foo"
    assert m.bytearray_size() == len("foo")


def test_capsule(capture):
    pytest.gc_collect()
    with capture:
        a = m.return_capsule_with_destructor()
        del a
        pytest.gc_collect()
    assert (
        capture.unordered
        == """
        creating capsule
        destructing capsule
    """
    )

    with capture:
        a = m.return_capsule_with_destructor_2()
        del a
        pytest.gc_collect()
    assert (
        capture.unordered
        == """
        creating capsule
        destructing capsule: 1234
    """
    )

    with capture:
        a = m.return_capsule_with_name_and_destructor()
        del a
        pytest.gc_collect()
    assert (
        capture.unordered
        == """
        created capsule (1234, 'pointer type description')
        destructing capsule (1234, 'pointer type description')
    """
    )


def test_accessors():
    class SubTestObject:
        attr_obj = 1
        attr_char = 2

    class TestObject:
        basic_attr = 1
        begin_end = [1, 2, 3]
        d = {"operator[object]": 1, "operator[char *]": 2}
        sub = SubTestObject()

        def func(self, x, *args):
            return self.basic_attr + x + sum(args)

    d = m.accessor_api(TestObject())
    assert d["basic_attr"] == 1
    assert d["begin_end"] == [1, 2, 3]
    assert d["operator[object]"] == 1
    assert d["operator[char *]"] == 2
    assert d["attr(object)"] == 1
    assert d["attr(char *)"] == 2
    assert d["missing_attr_ptr"] == "raised"
    assert d["missing_attr_chain"] == "raised"
    assert d["is_none"] is False
    assert d["operator()"] == 2
    assert d["operator*"] == 7
    assert d["implicit_list"] == [1, 2, 3]
    assert all(x in TestObject.__dict__ for x in d["implicit_dict"])

    assert m.tuple_accessor(tuple()) == (0, 1, 2)

    d = m.accessor_assignment()
    assert d["get"] == 0
    assert d["deferred_get"] == 0
    assert d["set"] == 1
    assert d["deferred_set"] == 1
    assert d["var"] == 99


def test_constructors():
    """C++ default and converting constructors are equivalent to type calls in Python"""
    types = [bytes, bytearray, str, bool, int, float, tuple, list, dict, set]
    expected = {t.__name__: t() for t in types}
    if env.PY2:
        # Note that bytes.__name__ == 'str' in Python 2.
        # pybind11::str is unicode even under Python 2.
        expected["bytes"] = bytes()
        expected["str"] = unicode()  # noqa: F821
    assert m.default_constructors() == expected

    data = {
        bytes: b"41",  # Currently no supported or working conversions.
        bytearray: bytearray(b"41"),
        str: 42,
        bool: "Not empty",
        int: "42",
        float: "+1e3",
        tuple: range(3),
        list: range(3),
        dict: [("two", 2), ("one", 1), ("three", 3)],
        set: [4, 4, 5, 6, 6, 6],
        memoryview: b"abc",
    }
    inputs = {k.__name__: v for k, v in data.items()}
    expected = {k.__name__: k(v) for k, v in data.items()}
    if env.PY2:  # Similar to the above. See comments above.
        inputs["bytes"] = b"41"
        inputs["str"] = 42
        expected["bytes"] = b"41"
        expected["str"] = u"42"

    assert m.converting_constructors(inputs) == expected
    assert m.cast_functions(inputs) == expected

    # Converting constructors and cast functions should just reference rather
    # than copy when no conversion is needed:
    noconv1 = m.converting_constructors(expected)
    for k in noconv1:
        assert noconv1[k] is expected[k]

    noconv2 = m.cast_functions(expected)
    for k in noconv2:
        assert noconv2[k] is expected[k]


def test_non_converting_constructors():
    non_converting_test_cases = [
        ("bytes", range(10)),
        ("none", 42),
        ("ellipsis", 42),
        ("type", 42),
    ]
    for t, v in non_converting_test_cases:
        for move in [True, False]:
            with pytest.raises(TypeError) as excinfo:
                m.nonconverting_constructor(t, v, move)
            expected_error = "Object of type '{}' is not an instance of '{}'".format(
                type(v).__name__, t
            )
            assert str(excinfo.value) == expected_error


def test_pybind11_str_raw_str():
    # specifically to exercise pybind11::str::raw_str
    cvt = m.convert_to_pybind11_str
    assert cvt(u"Str") == u"Str"
    assert cvt(b"Bytes") == u"Bytes" if env.PY2 else "b'Bytes'"
    assert cvt(None) == u"None"
    assert cvt(False) == u"False"
    assert cvt(True) == u"True"
    assert cvt(42) == u"42"
    assert cvt(2 ** 65) == u"36893488147419103232"
    assert cvt(-1.50) == u"-1.5"
    assert cvt(()) == u"()"
    assert cvt((18,)) == u"(18,)"
    assert cvt([]) == u"[]"
    assert cvt([28]) == u"[28]"
    assert cvt({}) == u"{}"
    assert cvt({3: 4}) == u"{3: 4}"
    assert cvt(set()) == u"set([])" if env.PY2 else "set()"
    assert cvt({3, 3}) == u"set([3])" if env.PY2 else "{3}"

    valid_orig = u"Ǳ"
    valid_utf8 = valid_orig.encode("utf-8")
    valid_cvt = cvt(valid_utf8)
    if hasattr(m, "PYBIND11_STR_LEGACY_PERMISSIVE"):
        assert valid_cvt is valid_utf8
    else:
        assert type(valid_cvt) is unicode if env.PY2 else str  # noqa: F821
        if env.PY2:
            assert valid_cvt == valid_orig
        else:
            assert valid_cvt == "b'\\xc7\\xb1'"

    malformed_utf8 = b"\x80"
    if hasattr(m, "PYBIND11_STR_LEGACY_PERMISSIVE"):
        assert cvt(malformed_utf8) is malformed_utf8
    else:
        if env.PY2:
            with pytest.raises(UnicodeDecodeError):
                cvt(malformed_utf8)
        else:
            malformed_cvt = cvt(malformed_utf8)
            assert type(malformed_cvt) is str
            assert malformed_cvt == "b'\\x80'"


def test_implicit_casting():
    """Tests implicit casting when assigning or appending to dicts and lists."""
    z = m.get_implicit_casting()
    assert z["d"] == {
        "char*_i1": "abc",
        "char*_i2": "abc",
        "char*_e": "abc",
        "char*_p": "abc",
        "str_i1": "str",
        "str_i2": "str1",
        "str_e": "str2",
        "str_p": "str3",
        "int_i1": 42,
        "int_i2": 42,
        "int_e": 43,
        "int_p": 44,
    }
    assert z["l"] == [3, 6, 9, 12, 15]


def test_print(capture):
    with capture:
        m.print_function()
    assert (
        capture
        == """
        Hello, World!
        1 2.0 three True -- multiple args
        *args-and-a-custom-separator
        no new line here -- next print
        flush
        py::print + str.format = this
    """
    )
    assert capture.stderr == "this goes to stderr"

    with pytest.raises(RuntimeError) as excinfo:
        m.print_failure()
    assert str(excinfo.value) == "Unable to convert call argument " + (
        "'1' of type 'UnregisteredType' to Python object"
        if debug_enabled
        else "to Python object (compile in debug mode for details)"
    )


def test_hash():
    class Hashable(object):
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return self.value

    class Unhashable(object):
        __hash__ = None

    assert m.hash_function(Hashable(42)) == 42
    with pytest.raises(TypeError):
        m.hash_function(Unhashable())


def test_number_protocol():
    for a, b in [(1, 1), (3, 5)]:
        li = [
            a == b,
            a != b,
            a < b,
            a <= b,
            a > b,
            a >= b,
            a + b,
            a - b,
            a * b,
            a / b,
            a | b,
            a & b,
            a ^ b,
            a >> b,
            a << b,
        ]
        assert m.test_number_protocol(a, b) == li


def test_list_slicing():
    li = list(range(100))
    assert li[::2] == m.test_list_slicing(li)


def test_issue2361():
    # See issue #2361
    assert m.issue2361_str_implicit_copy_none() == "None"
    with pytest.raises(TypeError) as excinfo:
        assert m.issue2361_dict_implicit_copy_none()
    assert "'NoneType' object is not iterable" in str(excinfo.value)


@pytest.mark.parametrize(
    "method, args, fmt, expected_view",
    [
        (m.test_memoryview_object, (b"red",), "B", b"red"),
        (m.test_memoryview_buffer_info, (b"green",), "B", b"green"),
        (m.test_memoryview_from_buffer, (False,), "h", [3, 1, 4, 1, 5]),
        (m.test_memoryview_from_buffer, (True,), "H", [2, 7, 1, 8]),
        (m.test_memoryview_from_buffer_nativeformat, (), "@i", [4, 7, 5]),
    ],
)
def test_memoryview(method, args, fmt, expected_view):
    view = method(*args)
    assert isinstance(view, memoryview)
    assert view.format == fmt
    if isinstance(expected_view, bytes) or not env.PY2:
        view_as_list = list(view)
    else:
        # Using max to pick non-zero byte (big-endian vs little-endian).
        view_as_list = [max(ord(c) for c in s) for s in view]
    assert view_as_list == list(expected_view)


@pytest.mark.xfail("env.PYPY", reason="getrefcount is not available")
@pytest.mark.parametrize(
    "method",
    [
        m.test_memoryview_object,
        m.test_memoryview_buffer_info,
    ],
)
def test_memoryview_refcount(method):
    buf = b"\x0a\x0b\x0c\x0d"
    ref_before = sys.getrefcount(buf)
    view = method(buf)
    ref_after = sys.getrefcount(buf)
    assert ref_before < ref_after
    assert list(view) == list(buf)


def test_memoryview_from_buffer_empty_shape():
    view = m.test_memoryview_from_buffer_empty_shape()
    assert isinstance(view, memoryview)
    assert view.format == "B"
    if env.PY2:
        # Python 2 behavior is weird, but Python 3 (the future) is fine.
        # PyPy3 has <memoryview, while CPython 2 has <memory
        assert bytes(view).startswith(b"<memory")
    else:
        assert bytes(view) == b""


def test_test_memoryview_from_buffer_invalid_strides():
    with pytest.raises(RuntimeError):
        m.test_memoryview_from_buffer_invalid_strides()


def test_test_memoryview_from_buffer_nullptr():
    if env.PY2:
        m.test_memoryview_from_buffer_nullptr()
    else:
        with pytest.raises(ValueError):
            m.test_memoryview_from_buffer_nullptr()


@pytest.mark.skipif("env.PY2")
def test_memoryview_from_memory():
    view = m.test_memoryview_from_memory()
    assert isinstance(view, memoryview)
    assert view.format == "B"
    assert bytes(view) == b"\xff\xe1\xab\x37"


def test_builtin_functions():
    assert m.get_len([i for i in range(42)]) == 42
    with pytest.raises(TypeError) as exc_info:
        m.get_len(i for i in range(42))
    assert str(exc_info.value) in [
        "object of type 'generator' has no len()",
        "'generator' has no length",
    ]  # PyPy


def test_isinstance_string_types():
    assert m.isinstance_pybind11_bytes(b"")
    assert not m.isinstance_pybind11_bytes(u"")

    assert m.isinstance_pybind11_str(u"")
    if hasattr(m, "PYBIND11_STR_LEGACY_PERMISSIVE"):
        assert m.isinstance_pybind11_str(b"")
    else:
        assert not m.isinstance_pybind11_str(b"")


def test_pass_bytes_or_unicode_to_string_types():
    assert m.pass_to_pybind11_bytes(b"Bytes") == 5
    with pytest.raises(TypeError):
        m.pass_to_pybind11_bytes(u"Str")

    if hasattr(m, "PYBIND11_STR_LEGACY_PERMISSIVE") or env.PY2:
        assert m.pass_to_pybind11_str(b"Bytes") == 5
    else:
        with pytest.raises(TypeError):
            m.pass_to_pybind11_str(b"Bytes")
    assert m.pass_to_pybind11_str(u"Str") == 3

    assert m.pass_to_std_string(b"Bytes") == 5
    assert m.pass_to_std_string(u"Str") == 3

    malformed_utf8 = b"\x80"
    if hasattr(m, "PYBIND11_STR_LEGACY_PERMISSIVE"):
        assert m.pass_to_pybind11_str(malformed_utf8) == 1
    elif env.PY2:
        with pytest.raises(UnicodeDecodeError):
            m.pass_to_pybind11_str(malformed_utf8)
    else:
        with pytest.raises(TypeError):
            m.pass_to_pybind11_str(malformed_utf8)


@pytest.mark.parametrize(
    "create_weakref, create_weakref_with_callback",
    [
        (m.weakref_from_handle, m.weakref_from_handle_and_function),
        (m.weakref_from_object, m.weakref_from_object_and_function),
    ],
)
def test_weakref(create_weakref, create_weakref_with_callback):
    from weakref import getweakrefcount

    # Apparently, you cannot weakly reference an object()
    class WeaklyReferenced(object):
        pass

    def callback(wr):
        # No `nonlocal` in Python 2
        callback.called = True

    obj = WeaklyReferenced()
    assert getweakrefcount(obj) == 0
    wr = create_weakref(obj)  # noqa: F841
    assert getweakrefcount(obj) == 1

    obj = WeaklyReferenced()
    assert getweakrefcount(obj) == 0
    callback.called = False
    wr = create_weakref_with_callback(obj, callback)  # noqa: F841
    assert getweakrefcount(obj) == 1
    assert not callback.called
    del obj
    pytest.gc_collect()
    assert callback.called