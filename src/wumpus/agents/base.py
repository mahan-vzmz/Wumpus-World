import abc
from typing import Any, Protocol

from wumpus.domain import Action, GameConfig
from wumpus.observation import Observation


class Agent(Protocol):
    """
    رابط مشترک (Interface) برای تمامی عامل‌های Wumpus World.
    
    این رابط تضمین می‌کند که همهٔ عامل‌ها با یک ساختار واحد توسط Runner اجرا شوند
    و هیچ‌کدام دسترسی مستقیمی به hidden_map نداشته باشند.
    """

    @abc.abstractmethod
    def reset(self, config: GameConfig, public_map_info: dict[str, Any], seed: int) -> None:
        """
        آماده‌سازی عامل برای شروع یک بازی جدید.
        
        Args:
            config: تنظیمات ثابت بازی (جان، امتیاز طلا، هزینه چاه و غیره).
            public_map_info: اطلاعات مجاز و عمومی نقشه (مثلاً موقعیت شروع).
            seed: بذر تصادفی برای تکرارپذیری تصمیمات عامل.
        """
        ...

    @abc.abstractmethod
    def choose_action(self, observation: Observation) -> Action:
        """
        انتخاب کنش بر اساس مشاهدهٔ فعلی.
        
        Args:
            observation: اطلاعاتی که عامل در این گام اجازهٔ دیدنش را دارد.
            
        Returns:
            کنش قانونی انتخاب‌شده.
        """
        ...

    def observe_transition(self, observation: Observation, action: Action, outcome: Any) -> None:
        """
        [اختیاری] اطلاع‌رسانی به عامل پس از انجام حرکت و دریافت نتیجهٔ آن.
        (برای عامل‌هایی که نیاز به یادگیری یا به‌روزرسانی حافظه در این مرحله دارند).
        """
        pass
