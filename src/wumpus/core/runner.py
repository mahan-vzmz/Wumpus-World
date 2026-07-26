import time
from dataclasses import dataclass

from wumpus.agents.base import Agent
from wumpus.core.domain import GameConfig, GameMap, GameState, Status
from wumpus.core.engine import init_state, step
from wumpus.core.observation import make_observation


@dataclass(frozen=True)
class RunResult:
    """خروجی اجرای یک نقشه (Episode) توسط یک عامل."""
    state: GameState
    error: str | None = None
    runtime_ms: float = 0.0

    @property
    def won(self) -> bool:
        return self.state.status == Status.WON


def run_episode(
    agent: Agent,
    game_map: GameMap,
    config: GameConfig,
    seed: int = 42,
) -> RunResult:
    """
    اجرای یک بازی کامل برای عامل مشخص‌شده روی نقشهٔ داده‌شده.
    
    این تابع خطاهای احتمالی درون عامل (مثل انتخاب کنش غیرقانونی یا Exception) را
    به‌صورت امن می‌گیرد تا کل آزمایش متوقف نشود.
    """
    start_time = time.perf_counter()

    def _elapsed() -> float:
        return (time.perf_counter() - start_time) * 1000.0

    def _fail(status: Status, exc: Exception) -> RunResult:
        state.status = status
        state.event_log.append(f"{status.value}: {exc}")
        return RunResult(state=state, error=str(exc), runtime_ms=_elapsed())

    state = init_state(game_map, config)

    # اطلاعات مجاز برای عامل (public_map_info)
    public_map_info = {
        "grid_size": config.grid_size,
        "exit_position": config.exit_position,
    }
    if getattr(agent, "requires_full_map", False):
        public_map_info["game_map"] = game_map

    # ۱. راه‌اندازی عامل — خطای اینجا خطای عامل است
    try:
        agent.reset(config, public_map_info, seed)
    except Exception as e:
        return _fail(Status.AGENT_ERROR, e)

    # SearchAgent can prove that no safe route exists before the first
    # transition.  Represent that outcome explicitly instead of asking an
    # empty plan to choose a fallback action.
    search_result = getattr(agent, "search_result", None)
    if search_result is not None and not search_result.solved:
        state.status = Status.NO_SOLUTION
        reason = search_result.reason or "no safe path to exit exists"
        state.event_log.append(f"NO_SOLUTION: {reason}")
        return RunResult(state=state, error=None, runtime_ms=_elapsed())

    # ۲. حلقهٔ اصلی بازی
    #
    # خطاها با scope باریک تفکیک می‌شوند: نقص موتور/ادراک به‌عنوان ENGINE_ERROR
    # ثبت می‌شود و به‌اشتباه به پای عامل نوشته نمی‌شود؛ خطای کد عامل یا کنش
    # غیرقانونی به‌صورت AGENT_ERROR ثبت می‌شود.
    while state.status == Status.RUNNING:
        # ساخت مشاهده (سمت موتور)
        try:
            obs = make_observation(game_map, config, state)
        except Exception as e:
            return _fail(Status.ENGINE_ERROR, e)

        # درخواست کنش از عامل (سمت عامل)
        try:
            action = agent.choose_action(obs)
        except Exception as e:
            return _fail(Status.AGENT_ERROR, e)

        # قرارداد عامل: کنش باید قانونی باشد
        if action not in obs.legal_actions:
            label = getattr(action, "value", action)
            return _fail(
                Status.AGENT_ERROR,
                ValueError(f"illegal action {label!r} is not in legal_actions"),
            )

        # اعمال کنش در موتور بازی (کنش از پیش قانونی است؛ خطای اینجا نقص موتور)
        try:
            state = step(game_map, config, state, action)
        except Exception as e:
            return _fail(Status.ENGINE_ERROR, e)

        # اطلاع‌رسانی نتیجه به عامل (اختیاری). خطای این callback نباید نتیجهٔ
        # قطعی‌شدهٔ بازی را پاک کند.
        try:
            agent.observe_transition(obs, action, state.status)
        except Exception as e:
            if state.status == Status.RUNNING:
                return _fail(Status.AGENT_ERROR, e)
            state.event_log.append(f"WARN observe_transition raised post-terminal: {e}")

    return RunResult(state=state, error=None, runtime_ms=_elapsed())
