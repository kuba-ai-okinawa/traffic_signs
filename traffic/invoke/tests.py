"""
Module with tests tasks for invoke
"""

import invoke


@invoke.task
def static_code_analysis(context):
    """
    Static code analysis task.
    Runs pylint, pycodestyle, xenon and eslint
    """

    python_code_directories = "./traffic ./scripts ./tests"

    context.run("pycodestyle {}".format(python_code_directories), echo=True)
    context.run("xenon . --max-absolute=B", echo=True)
    context.run("pylint {}".format(python_code_directories), echo=True)


@invoke.task
def unit_tests(context):
    """
    Unit tests task.
    Runs pytest on tests defined in commit stage directory
    """

    context.run("py.test ./tests/commit_stage", echo=True, pty=True)


@invoke.task
def commit_stage(context):
    """
    Commit stage task.
    Runs unit tests and static code analysis tasks
    """

    unit_tests(context)
    static_code_analysis(context)


@invoke.task
def acceptance_stage(context):
    """
    Acceptance stage task.
    Runs acceptance tests
    """

    context.run("py.test ./tests/acceptance_stage", echo=True, pty=True)
