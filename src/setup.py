from setuptools import setup

setup(
   name='neuromix',
   version='1.0',
   description='a module for creative neural design',
   author='Krubeal Danieli',
   author_email='dkirubel@gmail.com',
   packages=['neuromix'],  #same as name
   install_requires=['numpy', 'matplotlib'], #external packages as dependencies
)
