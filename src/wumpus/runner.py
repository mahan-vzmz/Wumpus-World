import time
from dataclasses import dataclass

from wumpus.agents.base import Agent
from wumpus.domain import GameConfig, GameMap, GameState, Status
from wumpus.engine import compute_diagnostic_score, compute_score, init_state, step
from wumpus.observation import make_observation


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

    # اطلاعات مجاز برای عامل (public_map_info)
    public_map_info = {
        "grid_size": config.grid_size,
        "exit_position": config.exit_position,
    }

    try:
        # ۱. راه‌اندازی عامل
        agent.reset(config, public_map_info, seed)

        # ۲. مقداردهی اولیه موتور بازی
        state = init_state(game_map, config)

        # ۳. حلقهٔ اصلی بازی
        while state.status == Status.RUNNING:
            # ساخت مشاهده برای عامل
            obs = make_observation(game_map, config, state)

            # درخواست کنش از عامل
            action = agent.choose_action(obs)

            # اعمال کنش در موتور بازی
            state = step(game_map, config, state, action)

            # اطلاع‌رسانی نتیجه به عامل (اختیاری برای عامل‌های آنلاین)
            # در اینجا outcome را می‌توان صرفاً همان وضعیت status فرستاد 
            agent.observe_transition(obs, action, state.status)

    except Exception as e:
        # اگر خطایی از سمت عامل یا قوانین موتور رخ دهد، بازی فوراً متوقف می‌شود
        elapsed = (time.perf_counter() - start_time) * 1000.0
        # ایجاد یک وضعیت ساختگی برای نشان‌دادن توقف ناگهانی
        if 'state' not in locals():
            # خطای پیش از مقداردهی state
            state = GameState(position=config.exit_position, health=0)
        
        return RunResult(state=state, error=str(e), runtime_ms=elapsed)

    elapsed = (time.perf_counter() - start_time) * 1000.0
    return RunResult(state=state, error=None, runtime_ms=elapsed)
