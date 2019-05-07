"""
Module with invoke tasks
"""

import invoke

import traffic.invoke.tests

# Create top level collection, populate it with tasks
namespace = invoke.Collection(__name__)
namespace.add_collection(traffic.invoke.tests, name="tests")
