from abc import abstractmethod, ABC
from typing import Optional, Dict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
import numpy as np
from yolov7obb.utils.general import linear, one_cycle


class ILRScheduler(ABC, _LRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            last_epoch: int = -1,
            warmup_epoch: Optional[int] = None,
            warmup_iterations: Optional[int] = None,
            verbose: Optional[bool] = False
    ) -> None:
        super().__init__(optimizer, last_epoch, verbose)
        self.warmup_epoch = warmup_epoch
        self.warmup_iterations = warmup_iterations

    @abstractmethod
    def warmup_step(self, global_iteration_number: int, epoch: int):
        pass

    @abstractmethod
    def step(self, epoch: Optional[int] = None):
        pass

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        pass


# Scheduler https://arxiv.org/pdf/1812.01187.pdf
class LRScheduler(ILRScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            epochs: Optional[int] = None,
            lr_final: Optional[float] = None,
            last_epoch: int = -1,
            warmup_epoch: Optional[int] = None,
            warmup_iterations: Optional[int] = None,
            verbose: Optional[bool] = False,
            linear_lr: Optional[bool] = False,
            warmup_bias_lr: Optional[float] = None,
            warmup_momentum: Optional[float] = None,
            momentum: Optional[float] = None
    ):
        self.linear_lr = linear_lr
        self.lr_final = lr_final
        self.epochs = epochs

        if self.linear_lr and self.lr_final and self.epochs:
            self.lf = linear(self.epochs, self.lr_final)  # linear
        elif self.lr_final and self.epochs:
            self.lf = one_cycle(1, self.lr_final, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = None

        if self.lf:
            self.scheduler = LambdaLR(optimizer, lr_lambda=self.lf)
        else:
            self.scheduler = None

        if warmup_epoch or warmup_iterations:
            assert warmup_bias_lr is not None, "Please set the parameter warmup_bias_lr"
            self.warmup_bias_lr = warmup_bias_lr
            assert warmup_momentum is not None, "Please set the parameter warmup_momentum"
            self.warmup_momentum = warmup_momentum
            assert momentum is not None, "Please set the parameter momentum"
            self.momentum = momentum

        super().__init__(
            optimizer=optimizer,
            last_epoch=last_epoch,
            warmup_epoch=warmup_epoch,
            warmup_iterations=warmup_iterations,
            verbose=verbose
        )

    def warmup_step(self, global_iteration_number: int, epoch: int):
        # Warmup
        assert global_iteration_number is not None, \
            f"Put global iteration number when call step for {self.__class__.__name__}"
        if global_iteration_number <= self.warmup_iterations:
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            # accumulate = max(1, np.interp(global_batch_number, xi, [1, nbs / total_batch_size]).round())
            for group_number, param in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0

                param['lr'] = np.interp(
                    global_iteration_number,
                    [0, self.warmup_iterations],
                    [self.warmup_bias_lr if group_number == 2 else 0.0, param['initial_lr'] * self.lf(epoch)]
                )
                if 'momentum' in param:
                    param['momentum'] = np.interp(
                        global_iteration_number,
                        [0, self.warmup_iterations],
                        [self.warmup_momentum, self.momentum]
                    )

    def step(self, epoch: Optional[int] = None):
        self.scheduler.step(epoch)

    def state_dict(self) -> Dict:
        state_dict = dict()
        state_dict["parent"] = super().state_dict()
        state_dict["lr_scheduler"] = self.scheduler.state_dict()
        state_dict["linear_lr"] = self.linear_lr
        state_dict["epochs"] = self.epochs
        state_dict["lr_final"] = self.lr_final
        state_dict["warmup_epoch"] = self.warmup_epoch
        state_dict["warmup_iterations"] = self.warmup_iterations
        state_dict["warmup_bias_lr"] = self.warmup_bias_lr
        state_dict["warmup_momentum"] = self.warmup_momentum
        state_dict["momentum"] = self.momentum
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict["parent"])
        self.linear_lr = state_dict["linear_lr"]
        if self.linear_lr:
            self.lf = linear(state_dict["epochs"], state_dict["lr_final"])  # linear
        else:
            self.lf = one_cycle(1, state_dict["lr_final"], state_dict["epochs"])  # cosine 1->hyp['lrf']
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.warmup_epoch = state_dict["warmup_epoch"]
        self.warmup_iterations = state_dict["warmup_iterations"]
        if self.warmup_epoch or self.warmup_iter_num:
            assert "warmup_bias_lr" in state_dict and state_dict["warmup_bias_lr"] is not None, \
                "warmup_bias_lr is not saved in the state or is None"
            self.warmup_bias_lr = state_dict["warmup_bias_lr"]
            assert "warmup_momentum" in state_dict and state_dict["warmup_momentum"] is not None, \
                "warmup_momentum is not saved in the state or is None"
            self.warmup_momentum = state_dict["warmup_momentum"]
            assert "momentum" in state_dict and state_dict["momentum"] is not None, \
                "momentum is not saved in the state or is None"
            self.momentum = state_dict["momentum"]