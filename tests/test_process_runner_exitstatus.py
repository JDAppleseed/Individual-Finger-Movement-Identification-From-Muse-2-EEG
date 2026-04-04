from app.process_runner import _normalize_exit_status, build_process_environment


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


def test_build_process_environment_overrides_source_id_per_run(monkeypatch) -> None:
    monkeypatch.setenv("LSL_SOURCE_ID", "stale")

    env_a = build_process_environment({"LSL_SOURCE_ID": "fresh-a"})
    env_b = build_process_environment({"LSL_SOURCE_ID": "fresh-b"})

    assert env_a.value("LSL_SOURCE_ID") == "fresh-a"
    assert env_b.value("LSL_SOURCE_ID") == "fresh-b"


def test_build_process_environment_can_remove_stale_source_id(monkeypatch) -> None:
    monkeypatch.setenv("LSL_SOURCE_ID", "stale")

    env = build_process_environment({"LSL_SOURCE_ID": None})

    assert env.contains("LSL_SOURCE_ID") is False
