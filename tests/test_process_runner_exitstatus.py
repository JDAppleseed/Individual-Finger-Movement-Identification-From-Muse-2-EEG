from app.process_runner import _normalize_exit_status


class FakeEnum:
    def __init__(self, value: int) -> None:
        self.value = value


class FakeIntable:
    def __init__(self, value: int) -> None:
        self._value = value

    def __int__(self) -> int:
        return self._value


class Weird:
    pass


def test_normalize_exit_status_enum_value() -> None:
    assert _normalize_exit_status(FakeEnum(0)) == 0
    assert _normalize_exit_status(FakeEnum(1)) == 1
    assert _normalize_exit_status(FakeEnum(2)) == 1


def test_normalize_exit_status_int_castable() -> None:
    assert _normalize_exit_status(FakeIntable(0)) == 0
    assert _normalize_exit_status(FakeIntable(1)) == 1


def test_normalize_exit_status_uncastable() -> None:
    assert _normalize_exit_status(Weird()) == 1


def test_normalize_exit_status_value_exception() -> None:
    class BadValue:
        @property
        def value(self) -> int:
            raise TypeError("boom")

    assert _normalize_exit_status(BadValue()) == 1
