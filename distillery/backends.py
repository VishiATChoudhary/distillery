"""Where the GPU job runs. A backend is anything with run(job, run_id) -> {"loss", "answers"}."""


class LocalBackend:
    """This machine (CUDA, Apple MPS, or CPU). Adapter saved to runs/<run_id>."""

    def run(self, job, run_id):
        from .student import train_and_generate

        return train_and_generate(**job, out_dir=f"runs/{run_id}")


class ModalBackend:
    """Rented cloud GPU via Modal. Adapter saved to Modal volume 'distillery-runs'."""

    def run(self, job, run_id):
        from .modal_app import app, train_remote

        with app.run():
            return train_remote.remote(job, run_id)


BACKENDS = {"local": LocalBackend, "modal": ModalBackend}
